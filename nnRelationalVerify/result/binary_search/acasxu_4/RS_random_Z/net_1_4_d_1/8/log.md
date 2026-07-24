## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_4.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 46.318565822800004


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495)
1: (-17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504)
2: (-17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372)
3: (-22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624)
4: (-20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585)

## BASE Result
execution time: IAR + LP analysis = 2.44 + 1.88 = 4.32 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -46.4113886, upper bound: 46.4113886


# Binary Search by BASE starts (time budget: 1195.68 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=55.00176239013672
rel_dist={3: [-46.41138857101316, 46.41138857101315]}

## Binary search (step 1) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=55.00176239013672
rel_dist={3: [-46.411366576283044, 46.411366576283044]}

## Binary search (step 2) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=55.00176239013672
rel_dist={3: [-46.41123857032521, 46.41123857032903]}

## Binary search (step 3) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=55.00176239013672
rel_dist={3: [-46.410039219938405, 46.4100392199384]}

## Binary search (step 4) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=55.00176239013672
rel_dist={3: [-46.40791696364876, 46.40791696367562]}

## Binary search (step 5) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=55.00176239013672
rel_dist={3: [-46.40672265181321, 46.40672265182802]}

## Binary search (step 6) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=55.00176239013672
rel_dist={3: [-46.40611311757378, 46.406113117581526]}

## Binary search (step 7) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=55.00176239013672
rel_dist={3: [-46.40580581063882, 46.40580581064276]}

## Binary search (step 8) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=55.00176239013672
rel_dist={3: [-46.405651916648736, 46.40565189290891]}

## Binary search (step 9) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=55.00176239013672
rel_dist={3: [-46.405574448472976, 46.4055744584641]}

## Binary search (step 10) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=55.00176239013672
rel_dist={3: [-46.40553548314551, 46.40553551008645]}

## Binary search (step 11) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=55.00176239013672
rel_dist={3: [-46.40551600158139, 46.405516036735335]}

## Binary search (step 12) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=55.00176239013672
rel_dist={3: [-46.40550626293548, 46.4055063016896]}

## Binary search (step 13) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=55.00176239013672
rel_dist={3: [-46.405501397303595, 46.405501397303524]}

## Binary search (step 14) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=55.00176239013672
rel_dist={3: [-46.40549902064443, 46.40549902136124]}

## Binary search (step 15) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=55.00176239013672
rel_dist={3: [-46.40549778989501, 46.4054978543421]}

## Binary search (step 16) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=55.00176239013672
rel_dist={3: [-46.40549734791982, 46.405497434778894]}

## Binary Search Result
Binary search time: 79.42 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1116.26 seconds

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4113864, upper bound: 46.4113864
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4113864, upper bound: 46.4113886
time: 0.88 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.05 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.05
Output dim: 3, lower bound: -46.4113864, upper bound: 46.4113864
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.05
Output dim: 3, lower bound: -46.4113864, upper bound: 46.4113886

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4064116, upper bound: 46.4064116
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4064116, upper bound: 46.4064116
time: 0.87 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4088155, upper bound: 46.4087867
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4088155, upper bound: 46.4088024
time: 0.85 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.15 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.15
Output dim: 3, lower bound: -46.4064116, upper bound: 46.4064116
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.15
Output dim: 3, lower bound: -46.4064116, upper bound: 46.4064116
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.15
Output dim: 3, lower bound: -46.4088155, upper bound: 46.4087867
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.15
Output dim: 3, lower bound: -46.4088155, upper bound: 46.4088024

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3848021, upper bound: 46.3879156
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3848021, upper bound: 46.3879156
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3232491, upper bound: 46.3232695
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3232491, upper bound: 46.3232695
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4086849, upper bound: 46.4068424
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4068424, upper bound: 46.4086852
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4014905, upper bound: 46.4014307
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4014905, upper bound: 46.4014307
time: 0.95 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.34 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 3, lower bound: -46.3848021, upper bound: 46.3879156
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 3, lower bound: -46.3848021, upper bound: 46.3879156
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 3, lower bound: -46.3232491, upper bound: 46.3232695
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 3, lower bound: -46.3232491, upper bound: 46.3232695
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 3, lower bound: -46.4086849, upper bound: 46.4068424
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 3, lower bound: -46.4068424, upper bound: 46.4086852
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 3, lower bound: -46.4014905, upper bound: 46.4014307
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 3, lower bound: -46.4014905, upper bound: 46.4014307

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3824796, upper bound: 46.3823987
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3823987, upper bound: 46.3843526
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3847779, upper bound: 46.3879156
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3848021, upper bound: 46.3847069
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3219423, upper bound: 46.3219423
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3219423, upper bound: 46.3219657
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3231456, upper bound: 46.3232626
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3232491, upper bound: 46.3232695
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4085261, upper bound: 46.4068424
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4086849, upper bound: 46.4064544
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4006179, upper bound: 46.3996713
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4006179, upper bound: 46.3996713
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3995390, upper bound: 46.3995390
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3995390, upper bound: 46.3998188
time: 1.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988345, upper bound: 46.3982864
time: 2.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3982968, upper bound: 46.3984495
time: 1.03 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 6.48 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -46.3824796, upper bound: 46.3823987
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -46.3823987, upper bound: 46.3843526
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -46.3847779, upper bound: 46.3879156
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -46.3848021, upper bound: 46.3847069
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -46.3219423, upper bound: 46.3219423
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -46.3219423, upper bound: 46.3219657
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -46.3231456, upper bound: 46.3232626
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -46.3232491, upper bound: 46.3232695
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -46.4085261, upper bound: 46.4068424
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -46.4086849, upper bound: 46.4064544
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -46.4006179, upper bound: 46.3996713
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -46.4006179, upper bound: 46.3996713
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -46.3995390, upper bound: 46.3995390
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -46.3995390, upper bound: 46.3998188
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -46.3988345, upper bound: 46.3982864
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.48
Output dim: 3, lower bound: -46.3982968, upper bound: 46.3984495

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3784173, upper bound: 46.3784173
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3785286, upper bound: 46.3784173
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3823987, upper bound: 46.3843526
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3823987, upper bound: 46.3823987
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3808382, upper bound: 46.3838832
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3809064, upper bound: 46.3838772
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3808382, upper bound: 46.3808382
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3809305, upper bound: 46.3809843
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3199331
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3219423, upper bound: 46.3219423
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3219423, upper bound: 46.3219423
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3093069, upper bound: 46.3093764
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3093069, upper bound: 46.3093764
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4079245, upper bound: 46.4063629
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4056892, upper bound: 46.4058525
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3999463, upper bound: 46.3996713
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3999082, upper bound: 46.3996713
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3990225, upper bound: 46.3990314
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3990225, upper bound: 46.3990182
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3732798, upper bound: 46.3732798
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3732798, upper bound: 46.3732798
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988974, upper bound: 46.3988380
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988974
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990807
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3965302, upper bound: 46.3965302
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3965720, upper bound: 46.3965302
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3697047, upper bound: 46.3697047
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3697047, upper bound: 46.3697047
time: 0.68 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3784173, upper bound: 46.3784173
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3785286, upper bound: 46.3784173
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3823987, upper bound: 46.3843526
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3823987, upper bound: 46.3823987
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3808382, upper bound: 46.3838832
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3809064, upper bound: 46.3838772
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3808382, upper bound: 46.3808382
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3809305, upper bound: 46.3809843
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3199331
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3219423, upper bound: 46.3219423
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3219423, upper bound: 46.3219423
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3093069, upper bound: 46.3093764
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3093069, upper bound: 46.3093764
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.4079245, upper bound: 46.4063629
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.4056892, upper bound: 46.4058525
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3999463, upper bound: 46.3996713
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3999082, upper bound: 46.3996713
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3990225, upper bound: 46.3990314
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3990225, upper bound: 46.3990182
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3732798, upper bound: 46.3732798
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3732798, upper bound: 46.3732798
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3988974, upper bound: 46.3988380
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988974
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990807
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3965302, upper bound: 46.3965302
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3965720, upper bound: 46.3965302
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3697047, upper bound: 46.3697047
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.29
Output dim: 3, lower bound: -46.3697047, upper bound: 46.3697047

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3785025, upper bound: 46.3784173
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3785286, upper bound: 46.3784173
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3814948, upper bound: 46.3830226
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3814948, upper bound: 46.3834945
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3784173, upper bound: 46.3784173
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3784173, upper bound: 46.3784173
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3793611, upper bound: 46.3819768
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3793611, upper bound: 46.3824300
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3785025, upper bound: 46.3784173
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3784173, upper bound: 46.3802834
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3784173, upper bound: 46.3784173
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3784173, upper bound: 46.3784173
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3793611, upper bound: 46.3795315
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3794381, upper bound: 46.3794748
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3199331
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4026449, upper bound: 46.4026449
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.4030636, upper bound: 46.4026513
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3975155, upper bound: 46.3975155
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3975155, upper bound: 46.3975155
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3990182, upper bound: 46.3990182
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3990182, upper bound: 46.3990182
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3990182, upper bound: 46.3990182
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3990182, upper bound: 46.3990182
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3969175, upper bound: 46.3971072
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3970523, upper bound: 46.3968493
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3990259, upper bound: 46.3990182
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3990259, upper bound: 46.3990182
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3724931, upper bound: 46.3724931
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3724931, upper bound: 46.3724931
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3724931, upper bound: 46.3724931
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3724931, upper bound: 46.3724931
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3990211, upper bound: 46.3988380
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990807
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990249
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3914883, upper bound: 46.3914883
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3914883, upper bound: 46.3914883
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3915484, upper bound: 46.3914883
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3915447, upper bound: 46.3914883
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3693728, upper bound: 46.3693728
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3693728, upper bound: 46.3693728
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3693728, upper bound: 46.3693728
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3693728, upper bound: 46.3693728
time: 0.77 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3785025, upper bound: 46.3784173
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3785286, upper bound: 46.3784173
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3814948, upper bound: 46.3830226
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3814948, upper bound: 46.3834945
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3784173, upper bound: 46.3784173
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3784173, upper bound: 46.3784173
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3793611, upper bound: 46.3819768
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3793611, upper bound: 46.3824300
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3785025, upper bound: 46.3784173
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3784173, upper bound: 46.3802834
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3784173, upper bound: 46.3784173
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3784173, upper bound: 46.3784173
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3793611, upper bound: 46.3795315
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3794381, upper bound: 46.3794748
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3199331
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.4026449, upper bound: 46.4026449
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.4030636, upper bound: 46.4026513
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3975155, upper bound: 46.3975155
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3975155, upper bound: 46.3975155
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3990182, upper bound: 46.3990182
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3990182, upper bound: 46.3990182
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3990182, upper bound: 46.3990182
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3990182, upper bound: 46.3990182
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3969175, upper bound: 46.3971072
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3970523, upper bound: 46.3968493
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3990259, upper bound: 46.3990182
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3990259, upper bound: 46.3990182
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3724931, upper bound: 46.3724931
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3724931, upper bound: 46.3724931
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3724931, upper bound: 46.3724931
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3724931, upper bound: 46.3724931
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3990211, upper bound: 46.3988380
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3988380
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3726379, upper bound: 46.3726379
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990807
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3988380, upper bound: 46.3990249
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3914883, upper bound: 46.3914883
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3914883, upper bound: 46.3914883
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3915484, upper bound: 46.3914883
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3915447, upper bound: 46.3914883
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3693728, upper bound: 46.3693728
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3693728, upper bound: 46.3693728
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3693728, upper bound: 46.3693728
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 3, lower bound: -46.3693728, upper bound: 46.3693728

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772934, upper bound: 46.3772216
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3795778
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3788836
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3800505
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3792259
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3795778
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3800505
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3788836
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3792259
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3773197, upper bound: 46.3772216
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3772216, upper bound: 46.3772216
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 3.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3172250, upper bound: 46.3172440
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3172440, upper bound: 46.3172250
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 46

Time for candidate selection: 3.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3197819, upper bound: 46.3198658
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3197819
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 3.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3196630, upper bound: 46.3198025
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198025, upper bound: 46.3196630
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 3.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3169477, upper bound: 46.3169477
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3169477, upper bound: 46.3169477
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 3.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3186288, upper bound: 46.3187712
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3187513, upper bound: 46.3188443
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 14

Time for candidate selection: 3.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3172250, upper bound: 46.3172440
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3172440, upper bound: 46.3172250
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=55.00176239013672
rel_dist={3: [-46.41138857101316, 46.41138857101315]}

## Binary search (step 1) starts
Candidate diff: 0.0312500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3247517, upper bound: 46.3247517
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3247517, upper bound: 46.3247517
time: 0.69 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.40 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.40
Output dim: 3, lower bound: -46.3247517, upper bound: 46.3247517
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.40
Output dim: 3, lower bound: -46.3247517, upper bound: 46.3247517

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3231856, upper bound: 46.3231745
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3231745, upper bound: 46.3231856
time: 0.78 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3121448, upper bound: 46.3121448
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3121448, upper bound: 46.3121448
time: 1.12 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 5.03 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.03
Output dim: 3, lower bound: -46.3231856, upper bound: 46.3231745
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.03
Output dim: 3, lower bound: -46.3231745, upper bound: 46.3231856
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 5.03
Output dim: 3, lower bound: -46.3121448, upper bound: 46.3121448
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 5.03
Output dim: 3, lower bound: -46.3121448, upper bound: 46.3121448

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.1152317, upper bound: 46.1152317
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.1153611, upper bound: 46.1152317
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.1152317, upper bound: 46.1153611
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.1152317, upper bound: 46.1153611
time: 0.64 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.57 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.57
Output dim: 3, lower bound: -46.1152317, upper bound: 46.1152317
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.57
Output dim: 3, lower bound: -46.1153611, upper bound: 46.1152317
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.57
Output dim: 3, lower bound: -46.1152317, upper bound: 46.1153611
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.57
Output dim: 3, lower bound: -46.1152317, upper bound: 46.1153611
Binary search (step 1): status=Status.VERIFIED, low=0.0312500, high=0.0625000, mid=0.0312500, abs_max=55.00176239013672
rel_dist={3: [-46.411366576283044, 46.411366576283044]}

## Binary search (step 2) starts
Candidate diff: 0.0468750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3247517, upper bound: 46.3247517
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3247517, upper bound: 46.3247517
time: 0.80 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.54 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.54
Output dim: 3, lower bound: -46.3247517, upper bound: 46.3247517
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.54
Output dim: 3, lower bound: -46.3247517, upper bound: 46.3247517

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.1240234, upper bound: 46.1240234
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.1240234, upper bound: 46.1240234
time: 0.65 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3247465, upper bound: 46.3247517
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3247517, upper bound: 46.3247465
time: 0.78 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.08 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 4.08
Output dim: 3, lower bound: -46.1240234, upper bound: 46.1240234
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 4.08
Output dim: 3, lower bound: -46.1240234, upper bound: 46.1240234
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 3, lower bound: -46.3247465, upper bound: 46.3247517
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 3, lower bound: -46.3247517, upper bound: 46.3247465

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3246368, upper bound: 46.3247501
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3247465, upper bound: 46.3247517
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3121390, upper bound: 46.3121448
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3121390, upper bound: 46.3121448
time: 0.80 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.86 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.86
Output dim: 3, lower bound: -46.3246368, upper bound: 46.3247501
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.86
Output dim: 3, lower bound: -46.3247465, upper bound: 46.3247517
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.86
Output dim: 3, lower bound: -46.3121390, upper bound: 46.3121448
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.86
Output dim: 3, lower bound: -46.3121390, upper bound: 46.3121448

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3235064, upper bound: 46.3235064
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3235064, upper bound: 46.3235064
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3235064, upper bound: 46.3235064
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3235064, upper bound: 46.3235064
time: 0.64 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.46 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.46
Output dim: 3, lower bound: -46.3235064, upper bound: 46.3235064
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.46
Output dim: 3, lower bound: -46.3235064, upper bound: 46.3235064
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.46
Output dim: 3, lower bound: -46.3235064, upper bound: 46.3235064
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.46
Output dim: 3, lower bound: -46.3235064, upper bound: 46.3235064

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
time: 0.64 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.66 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3199331, upper bound: 46.3198658
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.78 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.32 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3199331, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 3.05 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3177538, upper bound: 46.3178910
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3178910, upper bound: 46.3177538
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 14

Time for candidate selection: 3.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3172250, upper bound: 46.3172440
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3172440, upper bound: 46.3172250
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 3.08 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3196630, upper bound: 46.3198025
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198025, upper bound: 46.3196630
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 3.07 seconds

### Candidate
type: RSZ, layer: 3, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3191429, upper bound: 46.3191367
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3191367, upper bound: 46.3191429
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 2.98 seconds

### Candidate
type: RSZ, layer: 3, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3191429, upper bound: 46.3191377
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3191367, upper bound: 46.3191429
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 3.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3172250, upper bound: 46.3172440
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3172440, upper bound: 46.3172287
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 32

Time for candidate selection: 3.23 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3159851, upper bound: 46.3159851
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3159851, upper bound: 46.3159851
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 3.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3196630, upper bound: 46.3198025
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198025, upper bound: 46.3197553
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 3.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3192038, upper bound: 46.3187513
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3188039, upper bound: 46.3186739
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 3.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3149466, upper bound: 46.3148458
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3148458, upper bound: 46.3149466
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 3.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2801938, upper bound: 46.2801854
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2801938, upper bound: 46.2801854
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 3.23 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3196630, upper bound: 46.3198025
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198025, upper bound: 46.3196630
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 3.09 seconds

### Candidate
type: RSZ, layer: 3, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3197191, upper bound: 46.3198503
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198503, upper bound: 46.3198375
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 3.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3149466, upper bound: 46.3148458
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3148458, upper bound: 46.3149466
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 3.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198354, upper bound: 46.3198658
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198336
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 3.06 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3172250, upper bound: 46.3172440
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3172440, upper bound: 46.3172250
time: 1.08 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 7.22 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3177538, upper bound: 46.3178910
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3178910, upper bound: 46.3177538
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3172250, upper bound: 46.3172440
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3172440, upper bound: 46.3172250
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3196630, upper bound: 46.3198025
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3198025, upper bound: 46.3196630
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3191429, upper bound: 46.3191367
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3191367, upper bound: 46.3191429
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3191429, upper bound: 46.3191377
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3191367, upper bound: 46.3191429
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3172250, upper bound: 46.3172440
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3172440, upper bound: 46.3172287
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3159851, upper bound: 46.3159851
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3159851, upper bound: 46.3159851
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3196630, upper bound: 46.3198025
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3198025, upper bound: 46.3197553
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3192038, upper bound: 46.3187513
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3188039, upper bound: 46.3186739
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3149466, upper bound: 46.3148458
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3148458, upper bound: 46.3149466
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.2801938, upper bound: 46.2801854
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.2801938, upper bound: 46.2801854
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3196630, upper bound: 46.3198025
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3198025, upper bound: 46.3196630
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3197191, upper bound: 46.3198503
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3198503, upper bound: 46.3198375
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3149466, upper bound: 46.3148458
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3148458, upper bound: 46.3149466
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3198354, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198336
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3172250, upper bound: 46.3172440
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.22
Output dim: 3, lower bound: -46.3172440, upper bound: 46.3172250

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3185571, upper bound: 46.3186880
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3185571, upper bound: 46.3185571
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2799915, upper bound: 46.2801079
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2799915, upper bound: 46.2801079
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3176504, upper bound: 46.3177710
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3177059, upper bound: 46.3175663
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3190112, upper bound: 46.3191274
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3191211, upper bound: 46.3190311
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3190311, upper bound: 46.3191220
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3191274, upper bound: 46.3191223
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2767654, upper bound: 46.2767922
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2767654, upper bound: 46.2767922
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3147727, upper bound: 46.3147727
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3147727, upper bound: 46.3148762
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3176854, upper bound: 46.3177191
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3178325, upper bound: 46.3176580
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3191854, upper bound: 46.3187481
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3190143, upper bound: 46.3187071
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3187086, upper bound: 46.3186256
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3188007, upper bound: 46.3186707
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2800746, upper bound: 46.2799762
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2800746, upper bound: 46.2799762
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3159132, upper bound: 46.3159132
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3159132, upper bound: 46.3159132
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3190311, upper bound: 46.3191211
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3190112, upper bound: 46.3191274
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3196474, upper bound: 46.3197741
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3197869, upper bound: 46.3196389
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2800644, upper bound: 46.2801698
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2800878, upper bound: 46.2801698
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3159851, upper bound: 46.3159851
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3159851, upper bound: 46.3159851
time: 0.60 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 4.83 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.3185571, upper bound: 46.3186880
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.3185571, upper bound: 46.3185571
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.2799915, upper bound: 46.2801079
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.2799915, upper bound: 46.2801079
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.3176504, upper bound: 46.3177710
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.3177059, upper bound: 46.3175663
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.3190112, upper bound: 46.3191274
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.3191211, upper bound: 46.3190311
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.3190311, upper bound: 46.3191220
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.3191274, upper bound: 46.3191223
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.2767654, upper bound: 46.2767922
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.2767654, upper bound: 46.2767922
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.3147727, upper bound: 46.3147727
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.3147727, upper bound: 46.3148762
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.3176854, upper bound: 46.3177191
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.3178325, upper bound: 46.3176580
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.3191854, upper bound: 46.3187481
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.3190143, upper bound: 46.3187071
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.3187086, upper bound: 46.3186256
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.3188007, upper bound: 46.3186707
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.2800746, upper bound: 46.2799762
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.2800746, upper bound: 46.2799762
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.3159132, upper bound: 46.3159132
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.3159132, upper bound: 46.3159132
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.3190311, upper bound: 46.3191211
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.3190112, upper bound: 46.3191274
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.3196474, upper bound: 46.3197741
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.3197869, upper bound: 46.3196389
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.2800644, upper bound: 46.2801698
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.2800878, upper bound: 46.2801698
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.3159851, upper bound: 46.3159851
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.83
Output dim: 3, lower bound: -46.3159851, upper bound: 46.3159851

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2797128, upper bound: 46.2797128
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2797128, upper bound: 46.2797128
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3190112, upper bound: 46.3191274
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3190112, upper bound: 46.3190784
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2765350, upper bound: 46.2766403
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2765350, upper bound: 46.2766403
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3161670, upper bound: 46.3162807
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3161909, upper bound: 46.3162804
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3179375, upper bound: 46.3180199
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3180252, upper bound: 46.3180442
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3166282, upper bound: 46.3167733
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3172106, upper bound: 46.3166374
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3190143, upper bound: 46.3187035
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3187805, upper bound: 46.3187071
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3134564, upper bound: 46.3133772
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3133772, upper bound: 46.3133772
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3135031, upper bound: 46.3133772
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3133772, upper bound: 46.3133975
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 14

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3138124, upper bound: 46.3138020
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3138020, upper bound: 46.3138743
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3161670, upper bound: 46.3162893
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3161670, upper bound: 46.3162805
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3196275, upper bound: 46.3197600
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3196474, upper bound: 46.3197741
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3190905, upper bound: 46.3190851
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3191944, upper bound: 46.3190851
time: 0.65 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.90 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.2797128, upper bound: 46.2797128
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.2797128, upper bound: 46.2797128
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.3190112, upper bound: 46.3191274
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.3190112, upper bound: 46.3190784
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.2765350, upper bound: 46.2766403
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.2765350, upper bound: 46.2766403
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.3161670, upper bound: 46.3162807
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.3161909, upper bound: 46.3162804
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.3179375, upper bound: 46.3180199
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.3180252, upper bound: 46.3180442
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.3166282, upper bound: 46.3167733
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.3172106, upper bound: 46.3166374
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.3190143, upper bound: 46.3187035
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.3187805, upper bound: 46.3187071
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.3134564, upper bound: 46.3133772
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.3133772, upper bound: 46.3133772
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.3135031, upper bound: 46.3133772
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.3133772, upper bound: 46.3133975
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.3138124, upper bound: 46.3138020
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.3138020, upper bound: 46.3138743
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.3161670, upper bound: 46.3162893
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.3161670, upper bound: 46.3162805
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.3196275, upper bound: 46.3197600
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.3196474, upper bound: 46.3197741
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.3190905, upper bound: 46.3190851
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.90
Output dim: 3, lower bound: -46.3191944, upper bound: 46.3190851

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3159720, upper bound: 46.3159720
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3159720, upper bound: 46.3159720
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3138020, upper bound: 46.3138020
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3138020, upper bound: 46.3138566
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3165931, upper bound: 46.3166881
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3170648, upper bound: 46.3166405
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3185539, upper bound: 46.3186433
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3187089, upper bound: 46.3185539
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3168811, upper bound: 46.3168811
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3168811, upper bound: 46.3168811
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3176057, upper bound: 46.3178016
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3175858, upper bound: 46.3176872
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3138747, upper bound: 46.3138747
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3138747, upper bound: 46.3138747
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3139734, upper bound: 46.3138747
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3138747, upper bound: 46.3138747
time: 0.88 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 4.24 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 3, lower bound: -46.3159720, upper bound: 46.3159720
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 3, lower bound: -46.3159720, upper bound: 46.3159720
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 3, lower bound: -46.3138020, upper bound: 46.3138020
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 3, lower bound: -46.3138020, upper bound: 46.3138566
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 3, lower bound: -46.3165931, upper bound: 46.3166881
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 3, lower bound: -46.3170648, upper bound: 46.3166405
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 4.24
Output dim: 3, lower bound: -46.3185539, upper bound: 46.3186433
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 4.24
Output dim: 3, lower bound: -46.3187089, upper bound: 46.3185539
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 3, lower bound: -46.3168811, upper bound: 46.3168811
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 3, lower bound: -46.3168811, upper bound: 46.3168811
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 3, lower bound: -46.3176057, upper bound: 46.3178016
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 3, lower bound: -46.3175858, upper bound: 46.3176872
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 3, lower bound: -46.3138747, upper bound: 46.3138747
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 3, lower bound: -46.3138747, upper bound: 46.3138747
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 3, lower bound: -46.3139734, upper bound: 46.3138747
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 4.24
Output dim: 3, lower bound: -46.3138747, upper bound: 46.3138747

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3134795, upper bound: 46.3133034
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3133034, upper bound: 46.3133034
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3159558, upper bound: 46.3158565
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3158565, upper bound: 46.3158565
time: 0.79 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 5.27 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 5.27
Output dim: 3, lower bound: -46.3134795, upper bound: 46.3133034
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 5.27
Output dim: 3, lower bound: -46.3133034, upper bound: 46.3133034
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 5.27
Output dim: 3, lower bound: -46.3159558, upper bound: 46.3158565
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 5.27
Output dim: 3, lower bound: -46.3158565, upper bound: 46.3158565
Binary search (step 2): status=Status.VERIFIED, low=0.0468750, high=0.0625000, mid=0.0468750, abs_max=55.00176239013672
rel_dist={3: [-46.41138857101316, 46.41138857101316]}

## Binary search (step 3) starts
Candidate diff: 0.0546875


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3247517, upper bound: 46.3247517
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3247517, upper bound: 46.3247517
time: 0.72 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.46 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 3, lower bound: -46.3247517, upper bound: 46.3247517
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 3, lower bound: -46.3247517, upper bound: 46.3247517

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3121448, upper bound: 46.3121448
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3121448, upper bound: 46.3121448
time: 0.73 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3247517, upper bound: 46.3247501
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3247501, upper bound: 46.3247517
time: 0.79 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.49 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 4.49
Output dim: 3, lower bound: -46.3121448, upper bound: 46.3121448
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 4.49
Output dim: 3, lower bound: -46.3121448, upper bound: 46.3121448
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.49
Output dim: 3, lower bound: -46.3247517, upper bound: 46.3247501
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.49
Output dim: 3, lower bound: -46.3247501, upper bound: 46.3247517

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3235064, upper bound: 46.3235064
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3235064, upper bound: 46.3235064
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3121448, upper bound: 46.3121381
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3121448, upper bound: 46.3121381
time: 0.82 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.99 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.99
Output dim: 3, lower bound: -46.3235064, upper bound: 46.3235064
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.99
Output dim: 3, lower bound: -46.3235064, upper bound: 46.3235064
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.99
Output dim: 3, lower bound: -46.3121448, upper bound: 46.3121381
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.99
Output dim: 3, lower bound: -46.3121448, upper bound: 46.3121381

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3219423, upper bound: 46.3219423
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3219423, upper bound: 46.3219657
time: 0.68 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.32 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.32
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.32
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.32
Output dim: 3, lower bound: -46.3219423, upper bound: 46.3219423
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.32
Output dim: 3, lower bound: -46.3219423, upper bound: 46.3219657

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3199331
time: 0.59 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 5.01 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 3, lower bound: -46.3215262, upper bound: 46.3215262
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3199331

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3199331
time: 0.96 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.94 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.94
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3199331

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 3.02 seconds

### Candidate
type: RSZ, layer: 3, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3197191, upper bound: 46.3198503
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198503, upper bound: 46.3197935
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 3.08 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198336, upper bound: 46.3198658
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198421
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 3.05 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198336, upper bound: 46.3198658
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198336
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 3.05 seconds

### Candidate
type: RSZ, layer: 3, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3177538, upper bound: 46.3178910
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3177538, upper bound: 46.3177538
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 3.09 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3191654, upper bound: 46.3192663
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3192663, upper bound: 46.3191654
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 3.06 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198336, upper bound: 46.3198658
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198336
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 3.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3191654, upper bound: 46.3192663
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3191654, upper bound: 46.3191654
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 3.06 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3149466, upper bound: 46.3148458
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3148458, upper bound: 46.3149466
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 3.07 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3177538, upper bound: 46.3178910
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3178910, upper bound: 46.3177671
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 3.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3197819, upper bound: 46.3198658
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3197819
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 3.06 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2801664, upper bound: 46.2801701
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2801664, upper bound: 46.2801701
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 3.08 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3159851, upper bound: 46.3159851
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3159851, upper bound: 46.3159851
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34

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
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 3.08 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2801664, upper bound: 46.2802089
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2801664, upper bound: 46.2802089
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 42

Time for candidate selection: 3.06 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3159851, upper bound: 46.3159851
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3159851, upper bound: 46.3159851
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 3.09 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2801664, upper bound: 46.2802244
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2801664, upper bound: 46.2802244
time: 1.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 3.09 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3177746, upper bound: 46.3179584
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3178910, upper bound: 46.3177538
time: 0.64 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 8.12 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3197191, upper bound: 46.3198503
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3198503, upper bound: 46.3197935
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3198336, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198421
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3198336, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198336
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3177538, upper bound: 46.3178910
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3177538, upper bound: 46.3177538
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3191654, upper bound: 46.3192663
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3192663, upper bound: 46.3191654
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3198336, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3198336
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3191654, upper bound: 46.3192663
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3191654, upper bound: 46.3191654
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3149466, upper bound: 46.3148458
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3148458, upper bound: 46.3149466
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3177538, upper bound: 46.3178910
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3178910, upper bound: 46.3177671
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3197819, upper bound: 46.3198658
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3197819
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.2801664, upper bound: 46.2801701
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.2801664, upper bound: 46.2801701
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3159851, upper bound: 46.3159851
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3159851, upper bound: 46.3159851
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.2801664, upper bound: 46.2802089
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.2801664, upper bound: 46.2802089
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3159851, upper bound: 46.3159851
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3159851, upper bound: 46.3159851
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.2801664, upper bound: 46.2802244
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.2801664, upper bound: 46.2802244
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3177746, upper bound: 46.3179584
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.12
Output dim: 3, lower bound: -46.3178910, upper bound: 46.3177538

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3197191, upper bound: 46.3198503
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3197089, upper bound: 46.3197664
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3196474, upper bound: 46.3197295
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3196474, upper bound: 46.3196275
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3196430, upper bound: 46.3198025
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3196430, upper bound: 46.3196630
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3197819, upper bound: 46.3198421
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3197147
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3176697, upper bound: 46.3178910
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3178855, upper bound: 46.3177538
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2801664, upper bound: 46.2800644
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2801664, upper bound: 46.2800644
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3191654, upper bound: 46.3192663
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3191654, upper bound: 46.3191841
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3177551, upper bound: 46.3177551
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3178536, upper bound: 46.3177551
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3176697, upper bound: 46.3178910
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3178858, upper bound: 46.3177538
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3186288, upper bound: 46.3187250
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3187513, upper bound: 46.3186288
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3164470, upper bound: 46.3165110
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3164470, upper bound: 46.3164904
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3177551, upper bound: 46.3177551
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3178536, upper bound: 46.3177551
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2801664, upper bound: 46.2800644
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.2801664, upper bound: 46.2800644
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3186288, upper bound: 46.3186642
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3187513, upper bound: 46.3186288
time: 0.79 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.95 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.3197191, upper bound: 46.3198503
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.3197089, upper bound: 46.3197664
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.3196474, upper bound: 46.3197295
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.3196474, upper bound: 46.3196275
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.3196430, upper bound: 46.3198025
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.3196430, upper bound: 46.3196630
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.3197819, upper bound: 46.3198421
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.3198658, upper bound: 46.3197147
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.3176697, upper bound: 46.3178910
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.3178855, upper bound: 46.3177538
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.2801664, upper bound: 46.2800644
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.2801664, upper bound: 46.2800644
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.3191654, upper bound: 46.3192663
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.3191654, upper bound: 46.3191841
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.3177551, upper bound: 46.3177551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.3178536, upper bound: 46.3177551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.3176697, upper bound: 46.3178910
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.3178858, upper bound: 46.3177538
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.3186288, upper bound: 46.3187250
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.3187513, upper bound: 46.3186288
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.3164470, upper bound: 46.3165110
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.3164470, upper bound: 46.3164904
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.3177551, upper bound: 46.3177551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.3178536, upper bound: 46.3177551
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.2801664, upper bound: 46.2800644
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.2801664, upper bound: 46.2800644
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.3186288, upper bound: 46.3186642
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.95
Output dim: 3, lower bound: -46.3187513, upper bound: 46.3186288

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3190311, upper bound: 46.3191211
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3190112, upper bound: 46.3191274
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3159720, upper bound: 46.3159720
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -46.3159720, upper bound: 46.3159720
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3196474, upper bound: 46.3197295
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3196275, upper bound: 46.3196598
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3196275, upper bound: 46.3196275
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -46.3197869, upper bound: 46.3196275
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495
1: -17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504
2: -17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372
3: -22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624
4: -20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585

Time for backsubstitution: 2.41 seconds
Binary search (step 3): status=Status.UNKNOWN, low=0.0468750, high=0.0546875, mid=0.0546875, abs_max=55.00176239013672
rel_dist={3: [-46.41138857101316, 46.41138857101315]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 1116.59 seconds
