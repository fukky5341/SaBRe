## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 187.542370087


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746)
1: (-117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561)
2: (-169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212)
3: (-63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962)
4: (-188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.93 + 1.68 = 3.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -187.9182065, upper bound: 187.9182065

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6721943, upper bound: 187.8827013
time: 0.57 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6774721, upper bound: 187.6774721
time: 0.55 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.28 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.28
Output dim: 3, lower bound: -187.6721943, upper bound: 187.8827013
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.28
Output dim: 3, lower bound: -187.6774721, upper bound: 187.6774721

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -119.9573212, 108.7468872, -149.6440735, 126.7424088, -246.6997375, 258.3909607
1: -93.8448410, 101.7164536, -117.3338928, 118.4335785, -212.2784119, 219.0503540
2: -135.7941437, 113.6401825, -169.7016296, 131.6250763, -267.4192200, 283.3417969
3: -54.4526558, 137.4407501, -63.3496017, 169.2627869, -223.7154388, 200.7903442
4: -151.2667084, 114.1601410, -188.6523895, 133.4867859, -284.7534485, 302.8125305

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6721943, upper bound: 187.6721943
time: 0.61 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6721943, upper bound: 187.6774721
time: 0.54 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -182.1602936, 153.6605072, -149.6440735, 126.7424088, -308.9027100, 303.3045654
1: -143.0398865, 144.6755829, -117.3338928, 118.4335785, -261.4734497, 262.0094604
2: -206.5966339, 159.9870605, -169.7016296, 131.6250763, -338.2217102, 329.6886902
3: -78.3434219, 204.8266907, -63.3496017, 169.2627869, -247.6062012, 268.1763000
4: -229.7526093, 160.7959442, -188.6523895, 133.4867859, -363.2393799, 349.4483337

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6774721, upper bound: 187.6721943
time: 0.69 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6774721, upper bound: 187.6774721
time: 0.65 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.23 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.23
Output dim: 3, lower bound: -187.6721943, upper bound: 187.6721943
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.23
Output dim: 3, lower bound: -187.6721943, upper bound: 187.6774721
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.23
Output dim: 3, lower bound: -187.6774721, upper bound: 187.6721943
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.23
Output dim: 3, lower bound: -187.6774721, upper bound: 187.6774721

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -119.9573212, 108.7468872, -119.9573212, 108.7468872, -228.7041931, 228.7042084
1: -93.8448410, 101.7164536, -93.8448410, 101.7164536, -195.5612946, 195.5612946
2: -135.7941437, 113.6401825, -135.7941437, 113.6401825, -249.4343262, 249.4343262
3: -54.4526558, 137.4407501, -54.4526558, 137.4407501, -191.8934021, 191.8934021
4: -151.2667084, 114.1601410, -151.2667084, 114.1601410, -265.4268188, 265.4268188

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6673717, upper bound: 187.8769586
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6717293, upper bound: 187.8769586
time: 0.64 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -119.9573212, 108.7468872, -182.1602936, 153.6605072, -273.6178284, 290.9071655
1: -93.8448410, 101.7164536, -143.0398865, 144.6755829, -238.5203857, 244.7563477
2: -135.7941437, 113.6401825, -206.5966339, 159.9870605, -295.7811584, 320.2367859
3: -54.4526558, 137.4407501, -78.3434219, 204.8266907, -259.2793579, 215.7841492
4: -151.2667084, 114.1601410, -229.7526093, 160.7959442, -312.0625916, 343.9127197

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6673717, upper bound: 187.8823298
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6717293, upper bound: 187.8823298
time: 0.58 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -182.1602936, 153.6605072, -119.9573212, 108.7468872, -290.9071655, 273.6178284
1: -143.0398865, 144.6755829, -93.8448410, 101.7164536, -244.7563477, 238.5203857
2: -206.5966339, 159.9870605, -135.7941437, 113.6401825, -320.2368164, 295.7811584
3: -78.3434219, 204.8266907, -54.4526558, 137.4407501, -215.7841339, 259.2793579
4: -229.7526093, 160.7959442, -151.2667084, 114.1601410, -343.9127502, 312.0626221

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6311779, upper bound: 187.6309287
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6396430, upper bound: 187.6316936
time: 0.63 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -182.1602936, 153.6605072, -182.1602936, 153.6605072, -335.8208008, 335.8208008
1: -143.0398865, 144.6755829, -143.0398865, 144.6755829, -287.7154541, 287.7154541
2: -206.5966339, 159.9870605, -206.5966339, 159.9870605, -366.5836792, 366.5836792
3: -78.3434219, 204.8266907, -78.3434219, 204.8266907, -283.1701050, 283.1701050
4: -229.7526093, 160.7959442, -229.7526093, 160.7959442, -390.5485229, 390.5485535

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6311779, upper bound: 187.6309287
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6396430, upper bound: 187.6316936
time: 0.66 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.15 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 3, lower bound: -187.6673717, upper bound: 187.8769586
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 3, lower bound: -187.6717293, upper bound: 187.8769586
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 3, lower bound: -187.6673717, upper bound: 187.8823298
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 3, lower bound: -187.6717293, upper bound: 187.8823298
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 3, lower bound: -187.6311779, upper bound: 187.6309287
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 3, lower bound: -187.6396430, upper bound: 187.6316936
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 3, lower bound: -187.6311779, upper bound: 187.6309287
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 3, lower bound: -187.6396430, upper bound: 187.6316936

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -62.5989532, 68.6813965, -119.9573212, 108.7468872, -171.3457947, 188.6387177
1: -49.0822411, 64.2613220, -93.8448410, 101.7164536, -150.7986908, 158.1061707
2: -71.3749008, 71.9796066, -135.7941437, 113.6401825, -185.0150757, 207.7737122
3: -33.6786270, 77.7695847, -54.4526558, 137.4407501, -171.1193695, 132.2222443
4: -79.9356384, 71.8462219, -151.2667084, 114.1601410, -194.0957794, 223.1129303

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8726010, upper bound: 187.8726010
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8726010, upper bound: 187.8769586
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -115.7513199, 105.7508011, -119.9573212, 108.7468872, -224.4981689, 225.7081299
1: -90.5305862, 98.8567810, -93.8448410, 101.7164536, -192.2470398, 192.7015991
2: -131.0461578, 110.5165710, -135.7941437, 113.6401825, -244.6863403, 246.3106995
3: -52.9895172, 132.8755493, -54.4526558, 137.4407501, -190.4302673, 187.3281860
4: -145.9682770, 110.9683304, -151.2667084, 114.1601410, -260.1284180, 262.2350464

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8769586, upper bound: 187.8726010
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8769586, upper bound: 187.8769586
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -62.5989532, 68.6813965, -182.1602936, 153.6605072, -216.2594299, 250.8416901
1: -49.0822411, 64.2613220, -143.0398865, 144.6755829, -193.7578278, 207.3012085
2: -71.3749008, 71.9796066, -206.5966339, 159.9870605, -231.3619690, 278.5762329
3: -33.6786270, 77.7695847, -78.3434219, 204.8266907, -238.5053101, 156.1129913
4: -79.9356384, 71.8462219, -229.7526093, 160.7959442, -240.7315826, 301.5988159

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6279076, upper bound: 187.6291800
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6286725, upper bound: 187.6375376
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -115.7513199, 105.7508011, -182.1602936, 153.6605072, -269.4118347, 287.9111023
1: -90.5305862, 98.8567810, -143.0398865, 144.6755829, -235.2061462, 241.8966522
2: -131.0461578, 110.5165710, -206.5966339, 159.9870605, -291.0332031, 317.1132202
3: -52.9895172, 132.8755493, -78.3434219, 204.8266907, -257.8162231, 211.2189484
4: -145.9682770, 110.9683304, -229.7526093, 160.7959442, -306.7642212, 340.7209473

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301558, upper bound: 187.6312910
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6309206, upper bound: 187.6396486
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -149.2839355, 134.6893616, -119.9573212, 108.7468872, -258.0307922, 254.6466827
1: -116.9905624, 127.1221771, -93.8448410, 101.7164536, -218.7070160, 220.9670105
2: -169.2326202, 141.1778259, -135.7941437, 113.6401825, -282.8728027, 276.9718933
3: -69.0598907, 170.1712341, -54.4526558, 137.4407501, -206.5006256, 224.6238861
4: -188.5170746, 140.5437622, -151.2667084, 114.1601410, -302.6772156, 291.8104248

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291800, upper bound: 187.6279076
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312910, upper bound: 187.6301558
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -214.7231140, 185.9732056, -119.9573212, 108.7468872, -323.4700012, 305.9305115
1: -168.6910553, 175.9269562, -93.8448410, 101.7164536, -270.4074707, 269.7717285
2: -243.5518188, 193.9766388, -135.7941437, 113.6401825, -357.1919861, 329.7707214
3: -95.6573029, 241.0182190, -54.4526558, 137.4407501, -232.5220337, 295.4708862
4: -271.0773621, 193.9729767, -151.2667084, 114.1601410, -385.2374268, 345.2396240

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6375376, upper bound: 187.6286725
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6396486, upper bound: 187.6309206
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -149.2839355, 134.6893616, -182.1602936, 153.6605072, -302.9444580, 316.8496704
1: -116.9905624, 127.1221771, -143.0398865, 144.6755829, -261.6661072, 270.1620178
2: -169.2326202, 141.1778259, -206.5966339, 159.9870605, -329.2196655, 347.7744141
3: -69.0598907, 170.1712341, -78.3434219, 204.8266907, -273.8865967, 248.5146027
4: -188.5170746, 140.5437622, -229.7526093, 160.7959442, -349.3130188, 370.2963867

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6304130, upper bound: 187.6304130
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6304130, upper bound: 187.6309287
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -214.7231140, 185.9732056, -182.1602936, 153.6605072, -368.3836060, 368.1334839
1: -168.6910553, 175.9269562, -143.0398865, 144.6755829, -313.3666077, 318.9668579
2: -243.5518188, 193.9766388, -206.5966339, 159.9870605, -403.5388794, 400.5732727
3: -95.6573029, 241.0182190, -78.3434219, 204.8266907, -300.0774231, 319.3616333
4: -271.0773621, 193.9729767, -229.7526093, 160.7959442, -431.8732910, 423.7255554

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6388782, upper bound: 187.6311779
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6388782, upper bound: 187.6316936
time: 0.81 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.40 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 3, lower bound: -187.8726010, upper bound: 187.8726010
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 3, lower bound: -187.8726010, upper bound: 187.8769586
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 3, lower bound: -187.8769586, upper bound: 187.8726010
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 3, lower bound: -187.8769586, upper bound: 187.8769586
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 3, lower bound: -187.6279076, upper bound: 187.6291800
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 3, lower bound: -187.6286725, upper bound: 187.6375376
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 3, lower bound: -187.6301558, upper bound: 187.6312910
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 3, lower bound: -187.6309206, upper bound: 187.6396486
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 3, lower bound: -187.6291800, upper bound: 187.6279076
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 3, lower bound: -187.6312910, upper bound: 187.6301558
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 3, lower bound: -187.6375376, upper bound: 187.6286725
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 3, lower bound: -187.6396486, upper bound: 187.6309206
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 3, lower bound: -187.6304130, upper bound: 187.6304130
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 3, lower bound: -187.6304130, upper bound: 187.6309287
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 3, lower bound: -187.6388782, upper bound: 187.6311779
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 3, lower bound: -187.6388782, upper bound: 187.6316936

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -62.5989532, 68.6813965, -62.5989532, 68.6813965, -131.2803497, 131.2803497
1: -49.0822411, 64.2613220, -49.0822411, 64.2613220, -113.3435593, 113.3435516
2: -71.3749008, 71.9796066, -71.3749008, 71.9796066, -143.3545074, 143.3545074
3: -33.6786270, 77.7695847, -33.6786270, 77.7695847, -111.4482117, 111.4482117
4: -79.9356384, 71.8462219, -79.9356384, 71.8462219, -151.7818604, 151.7818604

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6267453, upper bound: 187.8219008
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6269385, upper bound: 187.6269385
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -62.5989532, 68.6813965, -115.7513199, 105.7508011, -168.3497620, 184.4326935
1: -49.0822411, 64.2613220, -90.5305862, 98.8567810, -147.9390106, 154.7919006
2: -71.3749008, 71.9796066, -131.0461578, 110.5165710, -181.8914795, 203.0257416
3: -33.6786270, 77.7695847, -52.9895172, 132.8755493, -166.5541687, 130.7590942
4: -79.9356384, 71.8462219, -145.9682770, 110.9683304, -190.9039612, 217.8144989

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6267453, upper bound: 187.8241489
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6269385, upper bound: 187.6291867
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -115.7513199, 105.7508011, -62.5989532, 68.6813965, -184.4327087, 168.3497620
1: -90.5305862, 98.8567810, -49.0822411, 64.2613220, -154.7919006, 147.9390259
2: -131.0461578, 110.5165710, -71.3749008, 71.9796066, -203.0257568, 181.8914795
3: -52.9895172, 132.8755493, -33.6786270, 77.7695847, -130.7590942, 166.5541534
4: -145.9682770, 110.9683304, -79.9356384, 71.8462219, -217.8144989, 190.9039612

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6267453, upper bound: 187.8219008
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291867, upper bound: 187.6290496
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -115.7513199, 105.7508011, -115.7513199, 105.7508011, -221.5021210, 221.5021210
1: -90.5305862, 98.8567810, -90.5305862, 98.8567810, -189.3873596, 189.3873444
2: -131.0461578, 110.5165710, -131.0461578, 110.5165710, -241.5627136, 241.5627136
3: -52.9895172, 132.8755493, -52.9895172, 132.8755493, -185.8650665, 185.8650665
4: -145.9682770, 110.9683304, -145.9682770, 110.9683304, -256.9366150, 256.9366150

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291626, upper bound: 187.8219008
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291867, upper bound: 187.6297159
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -62.5989532, 68.6813965, -149.2839355, 134.6893616, -197.2882843, 217.9653320
1: -49.0822411, 64.2613220, -116.9905624, 127.1221771, -176.2044220, 181.2518921
2: -71.3749008, 71.9796066, -169.2326202, 141.1778259, -212.5527344, 241.2122192
3: -33.6786270, 77.7695847, -69.0598907, 170.1712341, -203.8498535, 146.8294678
4: -79.9356384, 71.8462219, -188.5170746, 140.5437622, -220.4794006, 260.3632812

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6277144, upper bound: 187.6291800
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6277144, upper bound: 187.6291800
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -62.5989532, 68.6813965, -214.7231140, 185.9732056, -248.5721436, 283.4044495
1: -49.0822411, 64.2613220, -168.6910553, 175.9269562, -225.0092010, 232.9523773
2: -71.3749008, 71.9796066, -243.5518188, 193.9766388, -265.3515320, 315.5314331
3: -33.6786270, 77.7695847, -95.6573029, 241.0182190, -274.6967773, 172.7840881
4: -79.9356384, 71.8462219, -271.0773621, 193.9729767, -273.9085999, 342.9235840

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6284793, upper bound: 187.6375376
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6284793, upper bound: 187.6375376
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -115.7513199, 105.7508011, -149.2839355, 134.6893616, -250.4406281, 255.0347137
1: -90.5305862, 98.8567810, -116.9905624, 127.1221771, -217.6527710, 215.8473053
2: -131.0461578, 110.5165710, -169.2326202, 141.1778259, -272.2239380, 279.7492065
3: -52.9895172, 132.8755493, -69.0598907, 170.1712341, -223.1607513, 201.9354248
4: -145.9682770, 110.9683304, -188.5170746, 140.5437622, -286.5120239, 299.4854126

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6312910
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6312910
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -115.7513199, 105.7508011, -214.7231140, 185.9732056, -301.7245178, 320.4738464
1: -90.5305862, 98.8567810, -168.6910553, 175.9269562, -266.4575500, 267.5477905
2: -131.0461578, 110.5165710, -243.5518188, 193.9766388, -325.0227966, 354.0683899
3: -52.9895172, 132.8755493, -95.6573029, 241.0182190, -294.0077515, 227.6883850
4: -145.9682770, 110.9683304, -271.0773621, 193.9729767, -339.9412231, 382.0456848

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6308965, upper bound: 187.6396486
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6308965, upper bound: 187.6396486
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -149.2839355, 134.6893616, -62.5989532, 68.6813965, -217.9653320, 197.2882690
1: -116.9905624, 127.1221771, -49.0822411, 64.2613220, -181.2518921, 176.2044220
2: -169.2326202, 141.1778259, -71.3749008, 71.9796066, -241.2122192, 212.5527344
3: -69.0598907, 170.1712341, -33.6786270, 77.7695847, -146.8294678, 203.8498535
4: -188.5170746, 140.5437622, -79.9356384, 71.8462219, -260.3632812, 220.4794006

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259194, upper bound: 187.5273276
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259194, upper bound: 187.6279076
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -149.2839355, 134.6893616, -115.7513199, 105.7508011, -255.0347290, 250.4406433
1: -116.9905624, 127.1221771, -90.5305862, 98.8567810, -215.8473206, 217.6527710
2: -169.2326202, 141.1778259, -131.0461578, 110.5165710, -279.7492065, 272.2239380
3: -69.0598907, 170.1712341, -52.9895172, 132.8755493, -201.9354248, 223.1607513
4: -188.5170746, 140.5437622, -145.9682770, 110.9683304, -299.4854126, 286.5119934

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280304, upper bound: 187.5295757
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280304, upper bound: 187.6301558
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -214.7231140, 185.9732056, -62.5989532, 68.6813965, -283.4045105, 248.5721436
1: -168.6910553, 175.9269562, -49.0822411, 64.2613220, -232.9523773, 225.0091858
2: -243.5518188, 193.9766388, -71.3749008, 71.9796066, -315.5314331, 265.3515320
3: -95.6573029, 241.0182190, -33.6786270, 77.7695847, -172.7840881, 274.6967773
4: -271.0773621, 193.9729767, -79.9356384, 71.8462219, -342.9235840, 273.9085999

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6266799
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6286725
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -214.7231140, 185.9732056, -115.7513199, 105.7508011, -320.4738464, 301.7245178
1: -168.6910553, 175.9269562, -90.5305862, 98.8567810, -267.5477600, 266.4575195
2: -243.5518188, 193.9766388, -131.0461578, 110.5165710, -354.0683899, 325.0227661
3: -95.6573029, 241.0182190, -52.9895172, 132.8755493, -227.6883850, 294.0077515
4: -271.0773621, 193.9729767, -145.9682770, 110.9683304, -382.0456848, 339.9412231

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6323670, upper bound: 187.6289280
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6323670, upper bound: 187.6309206
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -149.2839355, 134.6893616, -149.2839355, 134.6893616, -283.9732361, 283.9732056
1: -116.9905624, 127.1221771, -116.9905624, 127.1221771, -244.1127319, 244.1127167
2: -169.2326202, 141.1778259, -169.2326202, 141.1778259, -310.4104614, 310.4104614
3: -69.0598907, 170.1712341, -69.0598907, 170.1712341, -239.2311096, 239.2311096
4: -188.5170746, 140.5437622, -188.5170746, 140.5437622, -329.0608215, 329.0608215

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6268885, upper bound: 187.5295517
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301490, upper bound: 187.6301317
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -149.2839355, 134.6893616, -214.7231140, 185.9732056, -335.2571411, 349.4124451
1: -116.9905624, 127.1221771, -168.6910553, 175.9269562, -292.9174500, 295.8131714
2: -169.2326202, 141.1778259, -243.5518188, 193.9766388, -363.2092590, 384.7296143
3: -69.0598907, 170.1712341, -95.6573029, 241.0182190, -310.0781250, 265.3165283
4: -188.5170746, 140.5437622, -271.0773621, 193.9729767, -382.4900208, 411.6211243

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6268885, upper bound: 187.5295757
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301490, upper bound: 187.6301558
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -214.7231140, 185.9732056, -149.2839355, 134.6893616, -349.4123840, 335.2571411
1: -168.6910553, 175.9269562, -116.9905624, 127.1221771, -295.8131714, 292.9174805
2: -243.5518188, 193.9766388, -169.2326202, 141.1778259, -384.7296143, 363.2092590
3: -95.6573029, 241.0182190, -69.0598907, 170.1712341, -265.3165283, 310.0780640
4: -271.0773621, 193.9729767, -188.5170746, 140.5437622, -411.6211243, 382.4900208

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312251, upper bound: 187.6289039
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6385066, upper bound: 187.6308966
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -214.7231140, 185.9732056, -214.7231140, 185.9732056, -400.6963196, 400.6963196
1: -168.6910553, 175.9269562, -168.6910553, 175.9269562, -344.6179504, 344.6179504
2: -243.5518188, 193.9766388, -243.5518188, 193.9766388, -437.5284424, 437.5284424
3: -95.6573029, 241.0182190, -95.6573029, 241.0182190, -336.2658691, 336.2658691
4: -271.0773621, 193.9729767, -271.0773621, 193.9729767, -465.0502930, 465.0503235

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312251, upper bound: 187.6289272
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6385066, upper bound: 187.6309206
time: 0.65 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.39 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6267453, upper bound: 187.8219008
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6269385, upper bound: 187.6269385
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6267453, upper bound: 187.8241489
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6269385, upper bound: 187.6291867
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6267453, upper bound: 187.8219008
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6291867, upper bound: 187.6290496
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6291626, upper bound: 187.8219008
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6291867, upper bound: 187.6297159
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6277144, upper bound: 187.6291800
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6277144, upper bound: 187.6291800
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6284793, upper bound: 187.6375376
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6284793, upper bound: 187.6375376
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6312910
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6312910
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6308965, upper bound: 187.6396486
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6308965, upper bound: 187.6396486
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6259194, upper bound: 187.5273276
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6259194, upper bound: 187.6279076
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6280304, upper bound: 187.5295757
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6280304, upper bound: 187.6301558
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6266799
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6286725
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6323670, upper bound: 187.6289280
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6323670, upper bound: 187.6309206
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6268885, upper bound: 187.5295517
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6301490, upper bound: 187.6301317
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6268885, upper bound: 187.5295757
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6301490, upper bound: 187.6301558
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6312251, upper bound: 187.6289039
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6385066, upper bound: 187.6308966
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6312251, upper bound: 187.6289272
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 3, lower bound: -187.6385066, upper bound: 187.6309206

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -40.9809875, 54.7791138, -62.5989532, 68.6813965, -109.6623840, 117.3780518
1: -32.1188278, 51.3675804, -49.0822411, 64.2613220, -96.3801346, 100.4498138
2: -47.0364113, 57.8066292, -71.3749008, 71.9796066, -119.0159912, 129.1815338
3: -27.2729225, 55.0080185, -33.6786270, 77.7695847, -105.0425110, 88.6866455
4: -53.0424118, 57.1621284, -79.9356384, 71.8462219, -124.8886337, 137.0977478

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6267453, upper bound: 187.6267453
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6267453, upper bound: 187.6269385
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -88.7503510, 95.0967102, -62.5989532, 68.6813965, -157.4317474, 157.6956329
1: -69.4899979, 90.1278381, -49.0822411, 64.2613220, -133.7513123, 139.2100830
2: -100.9360809, 99.7855835, -71.3749008, 71.9796066, -172.9156799, 171.1604919
3: -48.5418625, 106.7330246, -33.6786270, 77.7695847, -126.3114471, 140.4116516
4: -113.2336960, 98.6521301, -79.9356384, 71.8462219, -185.0799103, 178.5877686

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6269385, upper bound: 187.6267453
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6269385, upper bound: 187.6269385
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -40.9809875, 54.7791138, -115.7513199, 105.7508011, -146.7317810, 170.5303955
1: -32.1188278, 51.3675804, -90.5305862, 98.8567810, -130.9755859, 141.8981628
2: -47.0364113, 57.8066292, -131.0461578, 110.5165710, -157.5529785, 188.8527832
3: -27.2729225, 55.0080185, -52.9895172, 132.8755493, -160.1484680, 107.9975357
4: -53.0424118, 57.1621284, -145.9682770, 110.9683304, -164.0107422, 203.1304016

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6288563, upper bound: 187.6291626
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6288563, upper bound: 187.6291867
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -88.7503510, 95.0967102, -115.7513199, 105.7508011, -194.5011597, 210.8479767
1: -69.4899979, 90.1278381, -90.5305862, 98.8567810, -168.3467255, 180.6584167
2: -100.9360809, 99.7855835, -131.0461578, 110.5165710, -211.4526520, 230.8317413
3: -48.5418625, 106.7330246, -52.9895172, 132.8755493, -181.3689728, 159.7225342
4: -113.2336960, 98.6521301, -145.9682770, 110.9683304, -224.2020264, 244.6204071

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290496, upper bound: 187.6291626
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290496, upper bound: 187.6291867
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -89.4479752, 89.8819046, -62.5989532, 68.6813965, -158.1293488, 152.4808350
1: -69.8412399, 84.1715393, -49.0822411, 64.2613220, -134.1025696, 133.2537537
2: -101.2739258, 94.6007919, -71.3749008, 71.9796066, -173.2534943, 165.9756927
3: -45.8811989, 105.0105896, -33.6786270, 77.7695847, -123.6507874, 138.6892090
4: -113.0568390, 94.2403870, -79.9356384, 71.8462219, -184.9030609, 174.1760254

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291626, upper bound: 187.6288563
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291626, upper bound: 187.6290496
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -146.0465698, 134.7337646, -62.5989532, 68.6813965, -214.7279663, 197.3327179
1: -114.2232513, 127.2852173, -49.0822411, 64.2613220, -178.4845734, 176.3674622
2: -165.4884644, 141.3374176, -71.3749008, 71.9796066, -237.4680634, 212.7123108
3: -69.3965378, 166.8359833, -33.6786270, 77.7695847, -147.1661224, 200.5145874
4: -184.5755615, 140.5543518, -79.9356384, 71.8462219, -256.4217834, 220.4899902

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291867, upper bound: 187.6288563
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291867, upper bound: 187.6290496
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -89.4479752, 89.8819046, -115.7513199, 105.7508011, -195.1987762, 205.6332092
1: -69.8412399, 84.1715393, -90.5305862, 98.8567810, -168.6979675, 174.7021179
2: -101.2739258, 94.6007919, -131.0461578, 110.5165710, -211.7904816, 225.6469421
3: -45.8811989, 105.0105896, -52.9895172, 132.8755493, -178.7567444, 158.0001068
4: -113.0568390, 94.2403870, -145.9682770, 110.9683304, -224.0251770, 240.2086639

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312736, upper bound: 187.6295361
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312736, upper bound: 187.6297159
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -146.0465698, 134.7337646, -115.7513199, 105.7508011, -251.7973633, 250.4850769
1: -114.2232513, 127.2852173, -90.5305862, 98.8567810, -213.0800018, 217.8157959
2: -165.4884644, 141.3374176, -131.0461578, 110.5165710, -276.0050354, 272.3835754
3: -69.3965378, 166.8359833, -52.9895172, 132.8755493, -202.2720947, 219.8255005
4: -184.5755615, 140.5543518, -145.9682770, 110.9683304, -295.5438843, 286.5226135

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312977, upper bound: 187.6295361
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312977, upper bound: 187.6297159
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -40.9809875, 54.7791138, -149.2839355, 134.6893616, -175.6703491, 204.0630493
1: -32.1188278, 51.3675804, -116.9905624, 127.1221771, -159.2409973, 168.3581390
2: -47.0364113, 57.8066292, -169.2326202, 141.1778259, -188.2142181, 227.0392303
3: -27.2729225, 55.0080185, -69.0598907, 170.1712341, -197.4441528, 124.0679016
4: -53.0424118, 57.1621284, -188.5170746, 140.5437622, -193.5861511, 245.6791992

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6259194
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6291800
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -88.7503510, 95.0967102, -149.2839355, 134.6893616, -223.4397125, 244.3806458
1: -69.4899979, 90.1278381, -116.9905624, 127.1221771, -196.6121521, 207.1183929
2: -100.9360809, 99.7855835, -169.2326202, 141.1778259, -242.1139069, 269.0181885
3: -48.5418625, 106.7330246, -69.0598907, 170.1712341, -218.7130585, 175.7929077
4: -113.2336960, 98.6521301, -188.5170746, 140.5437622, -253.7774048, 287.1691589

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6259194
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6291800
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -40.9809875, 54.7791138, -214.6526947, 185.8925476, -226.8735352, 269.4317932
1: -32.1188278, 51.3675804, -168.6352844, 175.8498383, -207.9686584, 220.0028687
2: -47.0364113, 57.8066292, -243.4717102, 193.8917542, -240.9281464, 301.2783508
3: -27.2729225, 55.0080185, -95.6125031, 240.9383545, -268.2112732, 149.7953796
4: -53.0424118, 57.1621284, -270.9879456, 193.8897400, -246.9321136, 328.1500244

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6302560
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6291800
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -88.7503510, 95.0967102, -214.7231140, 185.9732056, -274.7235718, 309.8197937
1: -69.4899979, 90.1278381, -168.6910553, 175.9269562, -245.4169464, 258.8189087
2: -100.9360809, 99.7855835, -243.5518188, 193.9766388, -294.9127197, 343.3374023
3: -48.5418625, 106.7330246, -95.6573029, 241.0182190, -289.5600586, 201.6833038
4: -113.2336960, 98.6521301, -271.0773621, 193.9729767, -307.2066345, 369.7294617

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6259194
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6291800
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -89.4479752, 89.8819046, -149.2839355, 134.6893616, -224.1372833, 239.1658325
1: -69.8412399, 84.1715393, -116.9905624, 127.1221771, -196.9633942, 201.1620789
2: -101.2739258, 94.6007919, -169.2326202, 141.1778259, -242.4517365, 263.8334045
3: -45.8811989, 105.0105896, -69.0598907, 170.1712341, -216.0524139, 174.0704803
4: -113.0568390, 94.2403870, -188.5170746, 140.5437622, -253.6005859, 282.7574463

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6280304
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6287787
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -146.0465698, 134.7337646, -149.2839355, 134.6893616, -280.7358704, 284.0176392
1: -114.2232513, 127.2852173, -116.9905624, 127.1221771, -241.3454132, 244.2757874
2: -165.4884644, 141.3374176, -169.2326202, 141.1778259, -306.6662598, 310.5700378
3: -69.3965378, 166.8359833, -69.0598907, 170.1712341, -239.5677795, 235.8958740
4: -184.5755615, 140.5543518, -188.5170746, 140.5437622, -325.1193237, 329.0714111

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6280304
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6287787
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -89.4479752, 89.8819046, -214.7231140, 185.9732056, -275.4211731, 304.6050110
1: -69.8412399, 84.1715393, -168.6910553, 175.9269562, -245.7681885, 252.8625793
2: -101.2739258, 94.6007919, -243.5518188, 193.9766388, -295.2505493, 338.1526184
3: -45.8811989, 105.0105896, -95.6573029, 241.0182190, -286.8993835, 199.7381897
4: -113.0568390, 94.2403870, -271.0773621, 193.9729767, -307.0298157, 365.3177490

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6323670
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6333372
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -146.0465698, 134.7337646, -214.7231140, 185.9732056, -332.0197754, 349.4568176
1: -114.2232513, 127.2852173, -168.6910553, 175.9269562, -290.1501770, 295.9762573
2: -165.4884644, 141.3374176, -243.5518188, 193.9766388, -359.4650879, 384.8892212
3: -69.3965378, 166.8359833, -95.6573029, 241.0182190, -310.4147644, 261.6839600
4: -184.5755615, 140.5543518, -271.0773621, 193.9729767, -378.5485229, 411.6316833

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6280304
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6287787
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -88.6077271, 92.8103714, -62.5989532, 68.6813965, -157.2891235, 155.4093323
1: -69.5185242, 87.8645935, -49.0822411, 64.2613220, -133.7798462, 136.9468079
2: -100.9003143, 97.3328018, -71.3749008, 71.9796066, -172.8799133, 168.7077026
3: -47.0367928, 106.5991135, -33.6786270, 77.7695847, -124.8063812, 140.2777405
4: -113.0341187, 96.2283783, -79.9356384, 71.8462219, -184.8803406, 176.1640015

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259194, upper bound: 187.5271344
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259194, upper bound: 187.5273276
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -144.8092651, 131.6590424, -62.5989532, 68.6813965, -213.4906616, 194.2579956
1: -113.4549103, 124.2506790, -49.0822411, 64.2613220, -177.7162323, 173.3329163
2: -164.1941833, 138.0722961, -71.3749008, 71.9796066, -236.1737823, 209.4472046
3: -67.5839310, 165.3264618, -33.6786270, 77.7695847, -145.3535156, 199.0050964
4: -182.9160309, 137.3219604, -79.9356384, 71.8462219, -254.7622528, 217.2575836

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259194, upper bound: 187.6277144
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259194, upper bound: 187.6279076
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -88.6077271, 92.8103714, -115.7513199, 105.7508011, -194.3585205, 208.5616608
1: -69.5185242, 87.8645935, -90.5305862, 98.8567810, -168.3752747, 178.3951721
2: -100.9003143, 97.3328018, -131.0461578, 110.5165710, -211.4168854, 228.3789520
3: -47.0367928, 106.5991135, -52.9895172, 132.8755493, -179.9123383, 159.5886230
4: -113.0341187, 96.2283783, -145.9682770, 110.9683304, -224.0024414, 242.1966553

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259194, upper bound: 187.5295517
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259194, upper bound: 187.5295757
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -144.8092651, 131.6590424, -115.7513199, 105.7508011, -250.5600586, 247.4103546
1: -113.4549103, 124.2506790, -90.5305862, 98.8567810, -212.3116913, 214.7812653
2: -164.1941833, 138.0722961, -131.0461578, 110.5165710, -274.7107544, 269.1184387
3: -67.5839310, 165.3264618, -52.9895172, 132.8755493, -200.4594727, 218.3159790
4: -182.9160309, 137.3219604, -145.9682770, 110.9683304, -293.8843689, 283.2901917

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259194, upper bound: 187.6285483
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259194, upper bound: 187.6287846
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -143.9574738, 137.5898285, -62.5989532, 68.6813965, -212.6388702, 199.1383209
1: -112.9016113, 130.6958466, -49.0822411, 64.2613220, -177.1629333, 177.9627075
2: -163.7802124, 143.6897125, -71.3749008, 71.9796066, -235.7598267, 213.0910187
3: -70.0736237, 166.9585724, -33.6786270, 77.7695847, -145.9971161, 200.6371918
4: -182.9930725, 142.4557343, -79.9356384, 71.8462219, -254.8392944, 221.3930817

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6264867
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6265020
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -210.2807465, 183.0075378, -62.5989532, 68.6813965, -278.9621582, 245.6064911
1: -165.1631775, 173.1105804, -49.0822411, 64.2613220, -229.4244995, 222.1928253
2: -238.5625763, 190.9392853, -71.3749008, 71.9796066, -310.5421753, 262.3141479
3: -94.2064362, 236.2539673, -33.6786270, 77.7695847, -171.2803497, 269.9325562
4: -265.5162659, 190.8201752, -79.9356384, 71.8462219, -337.3624878, 270.7557983

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6284793
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6285039
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -143.9574738, 137.5898285, -115.7513199, 105.7508011, -249.7082825, 252.3827362
1: -112.9016113, 130.6958466, -90.5305862, 98.8567810, -211.7583618, 219.3861694
2: -163.7802124, 143.6897125, -131.0461578, 110.5165710, -274.2967834, 272.5823975
3: -70.0736237, 166.9585724, -52.9895172, 132.8755493, -200.9014130, 219.9480896
4: -182.9930725, 142.4557343, -145.9682770, 110.9683304, -293.9613953, 287.3131104

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6289039
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6289272
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -210.2807465, 183.0075378, -115.7513199, 105.7508011, -316.0315552, 298.7588501
1: -165.1631775, 173.1105804, -90.5305862, 98.8567810, -264.0198975, 263.6411438
2: -238.5625763, 190.9392853, -131.0461578, 110.5165710, -349.0791626, 321.9854126
3: -94.2064362, 236.2539673, -52.9895172, 132.8755493, -226.1846008, 289.2434692
4: -265.5162659, 190.8201752, -145.9682770, 110.9683304, -376.4845886, 336.7884216

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6291677
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6292015
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -88.6077271, 92.8103714, -149.2839355, 134.6893616, -223.2970734, 242.0942993
1: -69.5185242, 87.8645935, -116.9905624, 127.1221771, -196.6407013, 204.8551331
2: -100.9003143, 97.3328018, -169.2326202, 141.1778259, -242.0781403, 266.5654297
3: -47.0367928, 106.5991135, -69.0598907, 170.1712341, -217.2080231, 175.6589966
4: -113.0341187, 96.2283783, -188.5170746, 140.5437622, -253.5778809, 284.7454529

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5263085, upper bound: 187.5263085
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5263085, upper bound: 187.5295517
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -144.8092651, 131.6590424, -149.2839355, 134.6893616, -279.4986267, 280.9429626
1: -113.4549103, 124.2506790, -116.9905624, 127.1221771, -240.5770874, 241.2412415
2: -164.1941833, 138.0722961, -169.2326202, 141.1778259, -305.3720093, 307.3049316
3: -67.5839310, 165.3264618, -69.0598907, 170.1712341, -237.7551575, 234.3863525
4: -182.9160309, 137.3219604, -188.5170746, 140.5437622, -323.4597778, 325.8389587

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295690, upper bound: 187.6268885
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295690, upper bound: 187.6301317
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -88.6077271, 92.8103714, -214.7231140, 185.9732056, -274.5809326, 307.5334778
1: -69.5185242, 87.8645935, -168.6910553, 175.9269562, -245.4454803, 256.5556335
2: -100.9003143, 97.3328018, -243.5518188, 193.9766388, -294.8769531, 340.8845825
3: -47.0367928, 106.5991135, -95.6573029, 241.0182190, -288.0550232, 201.6009064
4: -113.0341187, 96.2283783, -271.0773621, 193.9729767, -307.0070801, 367.3056946

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6256607, upper bound: 187.5273276
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6256607, upper bound: 187.5295757
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -144.8092651, 131.6590424, -214.7231140, 185.9732056, -330.7824707, 346.3821411
1: -113.4549103, 124.2506790, -168.6910553, 175.9269562, -289.3818665, 292.9416809
2: -164.1941833, 138.0722961, -243.5518188, 193.9766388, -358.1708374, 381.6241150
3: -67.5839310, 165.3264618, -95.6573029, 241.0182190, -308.6020813, 260.2376709
4: -182.9160309, 137.3219604, -271.0773621, 193.9729767, -376.8889771, 408.3992615

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289213, upper bound: 187.6279076
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289213, upper bound: 187.6301558
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -143.9574738, 137.5898285, -149.2839355, 134.6893616, -278.6467896, 286.4543457
1: -112.9016113, 130.6958466, -116.9905624, 127.1221771, -240.0237579, 246.2988434
2: -163.7802124, 143.6897125, -169.2326202, 141.1778259, -304.9580383, 311.3924561
3: -70.0736237, 166.9585724, -69.0598907, 170.1712341, -238.5296021, 236.0184326
4: -182.9930725, 142.4557343, -188.5170746, 140.5437622, -323.5368347, 330.6277771

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5306451, upper bound: 187.6256607
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5306451, upper bound: 187.6289039
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -210.2807465, 183.0075378, -149.2839355, 134.6893616, -344.9700928, 332.2914429
1: -165.1631775, 173.1105804, -116.9905624, 127.1221771, -292.2853088, 290.1011353
2: -238.5625763, 190.9392853, -169.2326202, 141.1778259, -379.7403564, 360.1719055
3: -94.2064362, 236.2539673, -69.0598907, 170.1712341, -263.8127441, 305.3138428
4: -265.5162659, 190.8201752, -188.5170746, 140.5437622, -406.0600281, 379.3372498

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5379266, upper bound: 187.6276534
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5379266, upper bound: 187.6308965
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -143.9574738, 137.5898285, -214.7231140, 185.9732056, -329.9306641, 352.2167664
1: -112.9016113, 130.6958466, -168.6910553, 175.9269562, -288.8285217, 298.1502075
2: -163.7802124, 143.6897125, -243.5518188, 193.9766388, -357.7568359, 385.9594727
3: -70.0736237, 166.9585724, -95.6573029, 241.0182190, -309.4788818, 262.0924988
4: -182.9930725, 142.4557343, -271.0773621, 193.9729767, -376.9660339, 413.5330811

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6299974, upper bound: 187.6257908
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6299974, upper bound: 187.6257908
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -210.2807465, 183.0075378, -214.7231140, 185.9732056, -396.2539673, 397.7306213
1: -165.1631775, 173.1105804, -168.6910553, 175.9269562, -341.0900574, 341.8016052
2: -238.5625763, 190.9392853, -243.5518188, 193.9766388, -432.5392151, 434.4910889
3: -94.2064362, 236.2539673, -95.6573029, 241.0182190, -334.7620850, 331.2743225
4: -265.5162659, 190.8201752, -271.0773621, 193.9729767, -459.4892578, 461.8974915

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6372789, upper bound: 187.6278615
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6372789, upper bound: 187.6309206
time: 0.66 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.31 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6267453, upper bound: 187.6267453
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6267453, upper bound: 187.6269385
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6269385, upper bound: 187.6267453
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6269385, upper bound: 187.6269385
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6288563, upper bound: 187.6291626
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6288563, upper bound: 187.6291867
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6290496, upper bound: 187.6291626
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6290496, upper bound: 187.6291867
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6291626, upper bound: 187.6288563
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6291626, upper bound: 187.6290496
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6291867, upper bound: 187.6288563
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6291867, upper bound: 187.6290496
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6312736, upper bound: 187.6295361
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6312736, upper bound: 187.6297159
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6312977, upper bound: 187.6295361
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6312977, upper bound: 187.6297159
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6259194
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6291800
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6259194
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6291800
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6302560
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6291800
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6259194
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6291800
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6280304
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6287787
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6280304
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6287787
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6323670
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6333372
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6280304
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6287787
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6259194, upper bound: 187.5271344
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6259194, upper bound: 187.5273276
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6259194, upper bound: 187.6277144
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6259194, upper bound: 187.6279076
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6259194, upper bound: 187.5295517
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6259194, upper bound: 187.5295757
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6259194, upper bound: 187.6285483
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6259194, upper bound: 187.6287846
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6264867
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6265020
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6284793
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6285039
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6289039
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6289272
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6291677
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6292015
NS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.5263085, upper bound: 187.5263085
NS_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.5263085, upper bound: 187.5295517
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.5295690, upper bound: 187.6268885
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.5295690, upper bound: 187.6301317
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6256607, upper bound: 187.5273276
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6256607, upper bound: 187.5295757
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6289213, upper bound: 187.6279076
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6289213, upper bound: 187.6301558
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.5306451, upper bound: 187.6256607
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.5306451, upper bound: 187.6289039
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.5379266, upper bound: 187.6276534
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.5379266, upper bound: 187.6308965
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6299974, upper bound: 187.6257908
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6299974, upper bound: 187.6257908
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6372789, upper bound: 187.6278615
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.31
Output dim: 3, lower bound: -187.6372789, upper bound: 187.6309206

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -40.9809875, 54.7791138, -40.9809875, 54.7791138, -95.7601013, 95.7600937
1: -32.1188278, 51.3675804, -32.1188278, 51.3675804, -83.4863968, 83.4863968
2: -47.0364113, 57.8066292, -47.0364113, 57.8066292, -104.8430405, 104.8430405
3: -27.2729225, 55.0080185, -27.2729225, 55.0080185, -82.2809448, 82.2809448
4: -53.0424118, 57.1621284, -53.0424118, 57.1621284, -110.2045288, 110.2045135

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6250098, upper bound: 187.3877502
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6240025, upper bound: 187.8149799
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -40.9809875, 54.7791138, -88.7503510, 95.0967102, -136.0776825, 143.5294647
1: -32.1188278, 51.3675804, -69.4899979, 90.1278381, -122.2466507, 120.8575745
2: -47.0364113, 57.8066292, -100.9360809, 99.7855835, -146.8219757, 158.7427063
3: -27.2729225, 55.0080185, -48.5418625, 106.7330246, -134.0059357, 103.5112381
4: -53.0424118, 57.1621284, -113.2336960, 98.6521301, -151.6945496, 170.3957825

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6250098, upper bound: 187.3879327
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6240025, upper bound: 187.8151624
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -88.7503510, 95.0967102, -40.9809875, 54.7791138, -143.5294647, 136.0776825
1: -69.4899979, 90.1278381, -32.1188278, 51.3675804, -120.8575745, 122.2466583
2: -100.9360809, 99.7855835, -47.0364113, 57.8066292, -158.7427063, 146.8219910
3: -48.5418625, 106.7330246, -27.2729225, 55.0080185, -103.5112381, 134.0059509
4: -113.2336960, 98.6521301, -53.0424118, 57.1621284, -170.3957825, 151.6945496

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6250098, upper bound: 187.3720326
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6241850, upper bound: 187.6240023
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -88.7503510, 95.0967102, -88.7503510, 95.0967102, -183.8470459, 183.8470306
1: -69.4899979, 90.1278381, -69.4899979, 90.1278381, -159.6177979, 159.6177826
2: -100.9360809, 99.7855835, -100.9360809, 99.7855835, -200.7216644, 200.7216644
3: -48.5418625, 106.7330246, -48.5418625, 106.7330246, -155.2748566, 155.2748566
4: -113.2336960, 98.6521301, -113.2336960, 98.6521301, -211.8858032, 211.8858032

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6250098, upper bound: 187.3720998
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6241850, upper bound: 187.6240023
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -40.9809875, 54.7791138, -89.4479752, 89.8819046, -130.8628845, 144.2270508
1: -32.1188278, 51.3675804, -69.8412399, 84.1715393, -116.2903595, 121.2088165
2: -47.0364113, 57.8066292, -101.2739258, 94.6007919, -141.6372070, 159.0805511
3: -27.2729225, 55.0080185, -45.8811989, 105.0105896, -132.2835083, 100.8892136
4: -53.0424118, 57.1621284, -113.0568390, 94.2403870, -147.2828064, 170.2189484

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6278511, upper bound: 187.3900510
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6268437, upper bound: 187.8172807
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -40.9809875, 54.7791138, -146.0465698, 134.7337646, -175.7147522, 200.8256836
1: -32.1188278, 51.3675804, -114.2232513, 127.2852173, -159.4040375, 165.5908356
2: -47.0364113, 57.8066292, -165.4884644, 141.3374176, -188.3737946, 223.2950897
3: -27.2729225, 55.0080185, -69.3965378, 166.8359833, -194.1089020, 124.4045563
4: -53.0424118, 57.1621284, -184.5755615, 140.5543518, -193.5967407, 241.7376862

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6278511, upper bound: 187.3900510
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6268437, upper bound: 187.8172807
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -88.7503510, 95.0967102, -89.4479752, 89.8819046, -178.6322479, 184.5446320
1: -69.4899979, 90.1278381, -69.8412399, 84.1715393, -153.6614990, 159.9690552
2: -100.9360809, 99.7855835, -101.2739258, 94.6007919, -195.5368652, 201.0595093
3: -48.5418625, 106.7330246, -45.8811989, 105.0105896, -153.4188080, 152.6142273
4: -113.2336960, 98.6521301, -113.0568390, 94.2403870, -207.4740906, 211.7089539

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280475, upper bound: 187.3743333
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6270262, upper bound: 187.6263031
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -88.7503510, 95.0967102, -146.0465698, 134.7337646, -223.4841156, 241.1432648
1: -69.4899979, 90.1278381, -114.2232513, 127.2852173, -196.7751923, 204.3510895
2: -100.9360809, 99.7855835, -165.4884644, 141.3374176, -242.2734833, 265.2740479
3: -48.5418625, 106.7330246, -69.3965378, 166.8359833, -215.3645477, 176.1295624
4: -113.2336960, 98.6521301, -184.5755615, 140.5543518, -253.7879944, 283.2276611

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280475, upper bound: 187.3743334
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6270262, upper bound: 187.6263031
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -89.4479752, 89.8819046, -40.9809875, 54.7791138, -144.2270508, 130.8628845
1: -69.8412399, 84.1715393, -32.1188278, 51.3675804, -121.2088165, 116.2903595
2: -101.2739258, 94.6007919, -47.0364113, 57.8066292, -159.0805359, 141.6372070
3: -45.8811989, 105.0105896, -27.2729225, 55.0080185, -100.8892212, 132.2835083
4: -113.0568390, 94.2403870, -53.0424118, 57.1621284, -170.2189636, 147.2828064

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274424, upper bound: 187.4879346
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263038, upper bound: 187.8149799
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -89.4479752, 89.8819046, -88.7503510, 95.0967102, -184.5446472, 178.6322479
1: -69.8412399, 84.1715393, -69.4899979, 90.1278381, -159.9690552, 153.6614990
2: -101.2739258, 94.6007919, -100.9360809, 99.7855835, -201.0594940, 195.5368652
3: -45.8811989, 105.0105896, -48.5418625, 106.7330246, -152.6142120, 153.4187927
4: -113.0568390, 94.2403870, -113.2336960, 98.6521301, -211.7089691, 207.4740753

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274424, upper bound: 187.4881171
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263038, upper bound: 187.8151624
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -146.0465698, 134.7337646, -40.9809875, 54.7791138, -200.8256836, 175.7147522
1: -114.2232513, 127.2852173, -32.1188278, 51.3675804, -165.5908356, 159.4040375
2: -165.4884644, 141.3374176, -47.0364113, 57.8066292, -223.2950897, 188.3737946
3: -69.3965378, 166.8359833, -27.2729225, 55.0080185, -124.4045563, 194.1089020
4: -184.5755615, 140.5543518, -53.0424118, 57.1621284, -241.7376556, 193.5967407

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274664, upper bound: 187.4329129
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263032, upper bound: 187.6268437
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -146.0465698, 134.7337646, -88.7503510, 95.0967102, -241.1432800, 223.4841156
1: -114.2232513, 127.2852173, -69.4899979, 90.1278381, -204.3510590, 196.7751617
2: -165.4884644, 141.3374176, -100.9360809, 99.7855835, -265.2740479, 242.2734985
3: -69.3965378, 166.8359833, -48.5418625, 106.7330246, -176.1295624, 215.3645477
4: -184.5755615, 140.5543518, -113.2336960, 98.6521301, -283.2276306, 253.7880554

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274664, upper bound: 187.4329129
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263032, upper bound: 187.6268437
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -89.4479752, 89.8819046, -89.4479752, 89.8819046, -179.3298340, 179.3298492
1: -69.8412399, 84.1715393, -69.8412399, 84.1715393, -154.0127411, 154.0127411
2: -101.2739258, 94.6007919, -101.2739258, 94.6007919, -195.8747253, 195.8747101
3: -45.8811989, 105.0105896, -45.8811989, 105.0105896, -150.8917847, 150.8917847
4: -113.0568390, 94.2403870, -113.0568390, 94.2403870, -207.2972260, 207.2972260

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302836, upper bound: 187.4900308
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291451, upper bound: 187.8149336
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -89.4479752, 89.8819046, -146.0465698, 134.7337646, -224.1817169, 235.9284668
1: -69.8412399, 84.1715393, -114.2232513, 127.2852173, -197.1264496, 198.3947754
2: -101.2739258, 94.6007919, -165.4884644, 141.3374176, -242.6113129, 260.0892639
3: -45.8811989, 105.0105896, -69.3965378, 166.8359833, -212.7171631, 174.4071350
4: -113.0568390, 94.2403870, -184.5755615, 140.5543518, -253.6111755, 278.8159485

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302836, upper bound: 187.4900804
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291451, upper bound: 187.8150484
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -146.0465698, 134.7337646, -89.4479752, 89.8819046, -235.9284668, 224.1817017
1: -114.2232513, 127.2852173, -69.8412399, 84.1715393, -198.3947906, 197.1264343
2: -165.4884644, 141.3374176, -101.2739258, 94.6007919, -260.0892639, 242.6113281
3: -69.3965378, 166.8359833, -45.8811989, 105.0105896, -174.4071350, 212.7171631
4: -184.5755615, 140.5543518, -113.0568390, 94.2403870, -278.8159485, 253.6111755

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6303077, upper bound: 187.4349487
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291445, upper bound: 187.6273117
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -146.0465698, 134.7337646, -146.0465698, 134.7337646, -280.7803345, 280.7803040
1: -114.2232513, 127.2852173, -114.2232513, 127.2852173, -241.5084686, 241.5084534
2: -165.4884644, 141.3374176, -165.4884644, 141.3374176, -306.8258667, 306.8258667
3: -69.3965378, 166.8359833, -69.3965378, 166.8359833, -236.2325134, 236.2325134
4: -184.5755615, 140.5543518, -184.5755615, 140.5543518, -325.1298828, 325.1299133

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6303077, upper bound: 187.4349487
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291445, upper bound: 187.6273117
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -40.9809875, 54.7791138, -88.6077271, 92.8103714, -133.7913513, 143.3868256
1: -32.1188278, 51.3675804, -69.5185242, 87.8645935, -119.9834061, 120.8861008
2: -47.0364113, 57.8066292, -100.9003143, 97.3328018, -144.3691864, 158.7069397
3: -27.2729225, 55.0080185, -47.0367928, 106.5991135, -133.8720245, 102.0447998
4: -53.0424118, 57.1621284, -113.0341187, 96.2283783, -149.2707672, 170.1962280

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5055578, upper bound: 187.3872464
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5045504, upper bound: 187.8144761
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -40.9809875, 54.7791138, -144.8092651, 131.6590424, -172.6400299, 199.5883789
1: -32.1188278, 51.3675804, -113.4549103, 124.2506790, -156.3695068, 164.8224945
2: -47.0364113, 57.8066292, -164.1941833, 138.0722961, -185.1086884, 222.0008087
3: -27.2729225, 55.0080185, -67.5839310, 165.3264618, -192.5993805, 122.5919495
4: -53.0424118, 57.1621284, -182.9160309, 137.3219604, -190.3643494, 240.0781403

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5055578, upper bound: 187.3900602
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5045504, upper bound: 187.8172899
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -88.7503510, 95.0967102, -88.6077271, 92.8103714, -181.5606842, 183.7044373
1: -69.4899979, 90.1278381, -69.5185242, 87.8645935, -157.3545532, 159.6463470
2: -100.9360809, 99.7855835, -100.9003143, 97.3328018, -198.2688904, 200.6858978
3: -48.5418625, 106.7330246, -47.0367928, 106.5991135, -155.1409454, 153.7698212
4: -113.2336960, 98.6521301, -113.0341187, 96.2283783, -209.4620209, 211.6862488

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5057542, upper bound: 187.3715288
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5047329, upper bound: 187.6234985
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -88.7503510, 95.0967102, -144.8092651, 131.6590424, -220.4093933, 239.9059601
1: -69.4899979, 90.1278381, -113.4549103, 124.2506790, -193.7406616, 203.5827484
2: -100.9360809, 99.7855835, -164.1941833, 138.0722961, -239.0083771, 263.9797668
3: -48.5418625, 106.7330246, -67.5839310, 165.3264618, -213.8683167, 174.3169556
4: -113.2336960, 98.6521301, -182.9160309, 137.3219604, -250.5556030, 281.5681458

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5057542, upper bound: 187.3743426
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5047329, upper bound: 187.6263123
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -40.9809875, 54.7791138, -143.9036560, 137.5302429, -177.0198822, 198.6827545
1: -32.1188278, 51.3675804, -112.8598557, 130.6386871, -160.6643372, 164.2274323
2: -47.0364113, 57.8066292, -163.7188110, 143.6280823, -188.2590790, 221.5254059
3: -27.2729225, 55.0080185, -70.0413589, 166.8976746, -194.1705933, 123.0192642
4: -53.0424118, 57.1621284, -182.9244080, 142.3950195, -193.9698792, 240.0865326

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6247507, upper bound: 187.3897308
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6237433, upper bound: 187.8169605
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -40.9809875, 54.7791138, -210.2120361, 182.9277954, -223.9087830, 264.9910889
1: -32.1188278, 51.3675804, -165.1087494, 173.0343781, -205.1531982, 216.4763336
2: -47.0364113, 57.8066292, -238.4842834, 190.8552704, -237.8916626, 296.2909241
3: -27.2729225, 55.0080185, -94.1620255, 236.1759186, -263.4488525, 148.2920227
4: -53.0424118, 57.1621284, -265.4289551, 190.7378845, -243.7802887, 322.5910645

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6247507, upper bound: 187.3949111
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6237433, upper bound: 187.8221408
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -88.7503510, 95.0967102, -143.9574738, 137.5898285, -225.3655853, 239.0541840
1: -69.4899979, 90.1278381, -112.9016113, 130.6958466, -198.4430542, 203.0294189
2: -100.9360809, 99.7855835, -163.7802124, 143.6897125, -242.7204132, 263.5657959
3: -48.5418625, 106.7330246, -70.0736237, 166.9585724, -215.5003815, 174.8963318
4: -113.2336960, 98.6521301, -182.9930725, 142.4557343, -254.8316803, 281.6452026

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5055578, upper bound: 187.3718545
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5045504, upper bound: 187.6234985
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -88.7503510, 95.0967102, -210.2807465, 183.0075378, -271.7578735, 305.3774414
1: -69.4899979, 90.1278381, -165.1631775, 173.1105804, -242.6005707, 255.2909851
2: -100.9360809, 99.7855835, -238.5625763, 190.9392853, -291.8753662, 338.3481445
3: -48.5418625, 106.7330246, -94.2064362, 236.2539673, -284.7958069, 200.1795502
4: -113.2336960, 98.6521301, -265.5162659, 190.8201752, -304.0538635, 364.1683960

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6249471, upper bound: 187.3745005
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5045504, upper bound: 187.6234985
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -89.4479752, 89.8819046, -88.6077271, 92.8103714, -182.2582855, 178.4896240
1: -69.8412399, 84.1715393, -69.5185242, 87.8645935, -157.7057953, 153.6900330
2: -101.2739258, 94.6007919, -100.9003143, 97.3328018, -198.6067200, 195.5010986
3: -45.8811989, 105.0105896, -47.0367928, 106.5991135, -152.4803009, 152.0473785
4: -113.0568390, 94.2403870, -113.0341187, 96.2283783, -209.2852020, 207.2745056

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5079903, upper bound: 187.4874308
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5068518, upper bound: 187.8144761
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -89.4479752, 89.8819046, -144.8092651, 131.6590424, -221.1069946, 234.6911469
1: -69.8412399, 84.1715393, -113.4549103, 124.2506790, -194.0919037, 197.6264496
2: -101.2739258, 94.6007919, -164.1941833, 138.0722961, -239.3461914, 258.7949829
3: -45.8811989, 105.0105896, -67.5839310, 165.3264618, -211.2076569, 172.5945129
4: -113.0568390, 94.2403870, -182.9160309, 137.3219604, -250.3787994, 277.1564331

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5079903, upper bound: 187.4899905
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5068518, upper bound: 187.8144761
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -146.0465698, 134.7337646, -88.6077271, 92.8103714, -238.8569336, 223.3414917
1: -114.2232513, 127.2852173, -69.5185242, 87.8645935, -202.0878448, 196.8037415
2: -165.4884644, 141.3374176, -100.9003143, 97.3328018, -262.8212585, 242.2377319
3: -69.3965378, 166.8359833, -47.0367928, 106.5991135, -175.9956512, 213.8727570
4: -184.5755615, 140.5543518, -113.0341187, 96.2283783, -280.8039246, 253.5884705

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5080144, upper bound: 187.4324091
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5068512, upper bound: 187.6263398
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -146.0465698, 134.7337646, -144.8092651, 131.6590424, -277.7055969, 279.5430298
1: -114.2232513, 127.2852173, -113.4549103, 124.2506790, -238.4739380, 240.7401276
2: -165.4884644, 141.3374176, -164.1941833, 138.0722961, -303.5607605, 305.5316162
3: -69.3965378, 166.8359833, -67.5839310, 165.3264618, -234.7229919, 234.4199219
4: -184.5755615, 140.5543518, -182.9160309, 137.3219604, -321.8974609, 323.4703979

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5080144, upper bound: 187.4324091
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5068512, upper bound: 187.6269514
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -89.4479752, 89.8819046, -143.9574738, 137.5898285, -225.9231720, 233.8393860
1: -69.8412399, 84.1715393, -112.9016113, 130.6958466, -198.6280518, 197.0731201
2: -101.2739258, 94.6007919, -163.7802124, 143.6897125, -242.6262970, 258.3810120
3: -45.8811989, 105.0105896, -70.0736237, 166.9585724, -212.8397369, 172.9512329
4: -113.0568390, 94.2403870, -182.9930725, 142.4557343, -254.2246857, 277.2334595

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6271832, upper bound: 187.4899152
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260447, upper bound: 187.8169605
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -89.4479752, 89.8819046, -210.2807465, 183.0075378, -272.4555054, 300.1626587
1: -69.8412399, 84.1715393, -165.1631775, 173.1105804, -242.9517975, 249.3346863
2: -101.2739258, 94.6007919, -238.5625763, 190.9392853, -292.2131958, 333.1633606
3: -45.8811989, 105.0105896, -94.2064362, 236.2539673, -282.1351318, 198.2344513
4: -113.0568390, 94.2403870, -265.5162659, 190.8201752, -303.8770142, 359.7566528

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6271832, upper bound: 187.4946240
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260447, upper bound: 187.8169605
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -146.0465698, 134.7337646, -143.9574738, 137.5898285, -282.8959045, 278.6911926
1: -114.2232513, 127.2852173, -112.9016113, 130.6958466, -243.3010864, 240.1868286
2: -165.4884644, 141.3374176, -163.7802124, 143.6897125, -307.2244263, 305.1176147
3: -69.3965378, 166.8359833, -70.0736237, 166.9585724, -236.3551025, 234.8969727
4: -184.5755615, 140.5543518, -182.9930725, 142.4557343, -326.2515869, 323.5474243

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272073, upper bound: 187.4324091
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260054, upper bound: 187.6263398
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -146.0465698, 134.7337646, -210.2807465, 183.0075378, -329.0541077, 345.0145264
1: -114.2232513, 127.2852173, -165.1631775, 173.1105804, -287.3338318, 292.4483643
2: -165.4884644, 141.3374176, -238.5625763, 190.9392853, -356.4277344, 379.8999939
3: -69.3965378, 166.8359833, -94.2064362, 236.2539673, -305.6505127, 260.1801758
4: -184.5755615, 140.5543518, -265.5162659, 190.8201752, -375.3957520, 406.0706177

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272073, upper bound: 187.4348991
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260055, upper bound: 187.6269514
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -88.6077271, 92.8103714, -40.9809875, 54.7791138, -143.3868256, 133.7913513
1: -69.5185242, 87.8645935, -32.1188278, 51.3675804, -120.8861008, 119.9834061
2: -100.9003143, 97.3328018, -47.0364113, 57.8066292, -158.7069397, 144.3692017
3: -47.0367928, 106.5991135, -27.2729225, 55.0080185, -102.0447693, 133.8720245
4: -113.0341187, 96.2283783, -53.0424118, 57.1621284, -170.1962280, 149.2707672

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6241992, upper bound: 187.3640684
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6234987, upper bound: 187.5045501
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -88.6077271, 92.8103714, -88.7503510, 95.0967102, -183.7044373, 181.5606842
1: -69.5185242, 87.8645935, -69.4899979, 90.1278381, -159.6463470, 157.3545532
2: -100.9003143, 97.3328018, -100.9360809, 99.7855835, -200.6858978, 198.2688904
3: -47.0367928, 106.5991135, -48.5418625, 106.7330246, -153.7698212, 155.1409302
4: -113.0341187, 96.2283783, -113.2336960, 98.6521301, -211.6862488, 209.4620209

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6241992, upper bound: 187.3642509
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6234987, upper bound: 187.5047325
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -144.8092651, 131.6590424, -40.9809875, 54.7791138, -199.5883789, 172.6400299
1: -113.4549103, 124.2506790, -32.1188278, 51.3675804, -164.8224945, 156.3695068
2: -164.1941833, 138.0722961, -47.0364113, 57.8066292, -222.0007935, 185.1086884
3: -67.5839310, 165.3264618, -27.2729225, 55.0080185, -122.5919495, 192.5993805
4: -182.9160309, 137.3219604, -53.0424118, 57.1621284, -240.0781403, 190.3643494

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274597, upper bound: 187.4657145
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263125, upper bound: 187.6257306
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -144.8092651, 131.6590424, -88.7503510, 95.0967102, -239.9059448, 220.4093933
1: -113.4549103, 124.2506790, -69.4899979, 90.1278381, -203.5827484, 193.7406616
2: -164.1941833, 138.0722961, -100.9360809, 99.7855835, -263.9797668, 239.0083771
3: -67.5839310, 165.3264618, -48.5418625, 106.7330246, -174.3169556, 213.8682861
4: -182.9160309, 137.3219604, -113.2336960, 98.6521301, -281.5681763, 250.5556030

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274597, upper bound: 187.4658970
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263125, upper bound: 187.6259131
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -88.6077271, 92.8103714, -89.4479752, 89.8819046, -178.4896240, 182.2583160
1: -69.5185242, 87.8645935, -69.8412399, 84.1715393, -153.6900330, 157.7058105
2: -100.9003143, 97.3328018, -101.2739258, 94.6007919, -195.5010986, 198.6067047
3: -47.0367928, 106.5991135, -45.8811989, 105.0105896, -152.0473785, 152.4803009
4: -113.0341187, 96.2283783, -113.0568390, 94.2403870, -207.2745056, 209.2851868

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6270404, upper bound: 187.3663691
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263399, upper bound: 187.5068508
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -88.6077271, 92.8103714, -146.0465698, 134.7337646, -223.3414917, 238.8569336
1: -69.5185242, 87.8645935, -114.2232513, 127.2852173, -196.8037262, 202.0878296
2: -100.9003143, 97.3328018, -165.4884644, 141.3374176, -242.2377319, 262.8212585
3: -47.0367928, 106.5991135, -69.3965378, 166.8359833, -213.8727417, 175.9956512
4: -113.0341187, 96.2283783, -184.5755615, 140.5543518, -253.5884705, 280.8038940

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6270404, upper bound: 187.3663691
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263399, upper bound: 187.5068508
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -144.8092651, 131.6590424, -89.4479752, 89.8819046, -234.6911621, 221.1069946
1: -113.4549103, 124.2506790, -69.8412399, 84.1715393, -197.6264496, 194.0919189
2: -164.1941833, 138.0722961, -101.2739258, 94.6007919, -258.7949829, 239.3461914
3: -67.5839310, 165.3264618, -45.8811989, 105.0105896, -172.5945129, 211.2076569
4: -182.9160309, 137.3219604, -113.0568390, 94.2403870, -277.1564331, 250.3787842

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6303010, upper bound: 187.4680153
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291537, upper bound: 187.6263070
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -144.8092651, 131.6590424, -146.0465698, 134.7337646, -279.5430298, 277.7056274
1: -113.4549103, 124.2506790, -114.2232513, 127.2852173, -240.7401276, 238.4739380
2: -164.1941833, 138.0722961, -165.4884644, 141.3374176, -305.5316162, 303.5607605
3: -67.5839310, 165.3264618, -69.3965378, 166.8359833, -234.4199219, 234.7229919
4: -182.9160309, 137.3219604, -184.5755615, 140.5543518, -323.4703979, 321.8974609

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6303010, upper bound: 187.4680153
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291537, upper bound: 187.6264775
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -143.9036560, 137.5302429, -40.9809875, 54.7791138, -198.6827698, 177.0198975
1: -112.8598557, 130.6386871, -32.1188278, 51.3675804, -164.2274323, 160.6643219
2: -163.7188110, 143.6280823, -47.0364113, 57.8066292, -221.5254364, 188.2590790
3: -70.0413589, 166.8976746, -27.2729225, 55.0080185, -123.0192642, 194.1705933
4: -182.9244080, 142.3950195, -53.0424118, 57.1621284, -240.0865326, 193.9698639

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6285358, upper bound: 187.3947653
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6266275, upper bound: 187.6237432
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -143.9574738, 137.5898285, -88.7503510, 95.0967102, -239.0541840, 225.3655853
1: -112.9016113, 130.6958466, -69.4899979, 90.1278381, -203.0294342, 198.4430542
2: -163.7802124, 143.6897125, -100.9360809, 99.7855835, -263.5657654, 242.7203979
3: -70.0736237, 166.9585724, -48.5418625, 106.7330246, -174.8963318, 215.5003967
4: -182.9930725, 142.4557343, -113.2336960, 98.6521301, -281.6452026, 254.8317108

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6285358, upper bound: 187.3949487
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6266275, upper bound: 187.6237582
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -210.2120361, 182.9277954, -40.9809875, 54.7791138, -264.9911194, 223.9087830
1: -165.1087494, 173.0343781, -32.1188278, 51.3675804, -216.4763336, 205.1531982
2: -238.4842834, 190.8552704, -47.0364113, 57.8066292, -296.2909241, 237.8916626
3: -94.1620255, 236.1759186, -27.2729225, 55.0080185, -148.2920074, 263.4488220
4: -265.4289551, 190.7378845, -53.0424118, 57.1621284, -322.5910339, 243.7802887

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6358173, upper bound: 187.4947343
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6311633, upper bound: 187.6264760
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -210.2807465, 183.0075378, -88.7503510, 95.0967102, -305.3774414, 271.7578735
1: -165.1631775, 173.1105804, -69.4899979, 90.1278381, -255.2909698, 242.6005554
2: -238.5625763, 190.9392853, -100.9360809, 99.7855835, -338.3481445, 291.8753662
3: -94.2064362, 236.2539673, -48.5418625, 106.7330246, -200.1795502, 284.7958374
4: -265.5162659, 190.8201752, -113.2336960, 98.6521301, -364.1683960, 304.0538330

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6358173, upper bound: 187.4949191
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6311633, upper bound: 187.6264993
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -143.9574738, 137.5898285, -89.4479752, 89.8819046, -233.8393860, 225.9231720
1: -112.9016113, 130.6958466, -69.8412399, 84.1715393, -197.0731201, 198.6280212
2: -163.7802124, 143.6897125, -101.2739258, 94.6007919, -258.3810120, 242.6263123
3: -70.0736237, 166.9585724, -45.8811989, 105.0105896, -172.9512177, 212.8397522
4: -182.9930725, 142.4557343, -113.0568390, 94.2403870, -277.2334595, 254.2246857

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6313771, upper bound: 187.3970661
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6294688, upper bound: 187.6260440
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -143.9574738, 137.5898285, -146.0465698, 134.7337646, -278.6912231, 282.8958740
1: -112.9016113, 130.6958466, -114.2232513, 127.2852173, -240.1868134, 243.3010559
2: -163.7802124, 143.6897125, -165.4884644, 141.3374176, -305.1176147, 307.2244568
3: -70.0736237, 166.9585724, -69.3965378, 166.8359833, -234.8969727, 236.3550873
4: -182.9930725, 142.4557343, -184.5755615, 140.5543518, -323.5474243, 326.2516174

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6313771, upper bound: 187.3970662
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6294688, upper bound: 187.6260439
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -210.2807465, 183.0075378, -89.4479752, 89.8819046, -300.1626587, 272.4555054
1: -165.1631775, 173.1105804, -69.8412399, 84.1715393, -249.3346863, 242.9517975
2: -238.5625763, 190.9392853, -101.2739258, 94.6007919, -333.1633606, 292.2131958
3: -94.2064362, 236.2539673, -45.8811989, 105.0105896, -198.2344360, 282.1351318
4: -265.5162659, 190.8201752, -113.0568390, 94.2403870, -359.7566528, 303.8770142

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6386586, upper bound: 187.4969826
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6340045, upper bound: 187.6269505
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -210.2807465, 183.0075378, -146.0465698, 134.7337646, -345.0144653, 329.0541077
1: -165.1631775, 173.1105804, -114.2232513, 127.2852173, -292.4483643, 287.3338318
2: -238.5625763, 190.9392853, -165.4884644, 141.3374176, -379.8999939, 356.4277344
3: -94.2064362, 236.2539673, -69.3965378, 166.8359833, -260.1801758, 305.6505127
4: -265.5162659, 190.8201752, -184.5755615, 140.5543518, -406.0706177, 375.3957520

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6386586, upper bound: 187.4969978
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6340046, upper bound: 187.6269785
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -144.8092651, 131.6590424, -88.6077271, 92.8103714, -237.6196289, 220.2667694
1: -113.4549103, 124.2506790, -69.5185242, 87.8645935, -201.3195038, 193.7691956
2: -164.1941833, 138.0722961, -100.9003143, 97.3328018, -261.5269775, 238.9726105
3: -67.5839310, 165.3264618, -47.0367928, 106.5991135, -174.1830444, 212.3632507
4: -182.9160309, 137.3219604, -113.0341187, 96.2283783, -279.1444092, 250.3560791

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5080076, upper bound: 187.4652107
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5068604, upper bound: 187.6252268
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -144.8092651, 131.6590424, -144.8092651, 131.6590424, -276.4683228, 276.4683228
1: -113.4549103, 124.2506790, -113.4549103, 124.2506790, -237.7055969, 237.7055969
2: -164.1941833, 138.0722961, -164.1941833, 138.0722961, -302.2664795, 302.2664795
3: -67.5839310, 165.3264618, -67.5839310, 165.3264618, -232.9104004, 232.9104004
4: -182.9160309, 137.3219604, -182.9160309, 137.3219604, -320.2379456, 320.2379150

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5080077, upper bound: 187.4680245
time: 0.96 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5068604, upper bound: 187.6259756
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -88.6077271, 92.8103714, -143.9574738, 137.5898285, -225.2692871, 236.7678528
1: -69.5185242, 87.8645935, -112.9016113, 130.6958466, -198.5025024, 200.7661743
2: -100.9003143, 97.3328018, -163.7802124, 143.6897125, -242.7417755, 261.1130066
3: -47.0367928, 106.5991135, -70.0736237, 166.9585724, -213.9953613, 174.8139343
4: -113.0341187, 96.2283783, -182.9930725, 142.4557343, -254.6831818, 279.2214355

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6239400, upper bound: 187.3641368
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6232395, upper bound: 187.5046185
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -88.6077271, 92.8103714, -210.2807465, 183.0075378, -271.6152344, 303.0911255
1: -69.5185242, 87.8645935, -165.1631775, 173.1105804, -242.6291046, 253.0277100
2: -100.9003143, 97.3328018, -238.5625763, 190.9392853, -291.8395996, 335.8953857
3: -47.0367928, 106.5991135, -94.2064362, 236.2539673, -283.2907715, 200.0971680
4: -113.0341187, 96.2283783, -265.5162659, 190.8201752, -303.8543091, 361.7446289

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6239400, upper bound: 187.3663691
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6232395, upper bound: 187.5068508
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -144.8092651, 131.6590424, -143.9574738, 137.5898285, -281.7225647, 275.6165161
1: -113.4549103, 124.2506790, -112.9016113, 130.6958466, -242.5797119, 237.1522827
2: -164.1941833, 138.0722961, -163.7802124, 143.6897125, -306.0038452, 301.8525085
3: -67.5839310, 165.3264618, -70.0736237, 166.9585724, -234.5424957, 233.4506531
4: -182.9160309, 137.3219604, -182.9930725, 142.4557343, -324.6633911, 320.3149719

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272006, upper bound: 187.4659287
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260533, upper bound: 187.6257991
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -144.8092651, 131.6590424, -210.2807465, 183.0075378, -327.8168030, 341.9397888
1: -113.4549103, 124.2506790, -165.1631775, 173.1105804, -286.5654907, 289.4138489
2: -164.1941833, 138.0722961, -238.5625763, 190.9392853, -355.1334839, 376.6348877
3: -67.5839310, 165.3264618, -94.2064362, 236.2539673, -303.8377991, 258.7338257
4: -182.9160309, 137.3219604, -265.5162659, 190.8201752, -373.7362061, 402.8381958

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272006, upper bound: 187.4684190
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260533, upper bound: 187.6264775
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -143.9574738, 137.5898285, -88.6077271, 92.8103714, -236.7678223, 225.2692566
1: -112.9016113, 130.6958466, -69.5185242, 87.8645935, -200.7661743, 198.5025177
2: -163.7802124, 143.6897125, -100.9003143, 97.3328018, -261.1130066, 242.7417297
3: -70.0736237, 166.9585724, -47.0367928, 106.5991135, -174.8139496, 213.9953461
4: -182.9930725, 142.4557343, -113.0341187, 96.2283783, -279.2214355, 254.6831970

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5090837, upper bound: 187.3942621
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5071755, upper bound: 187.6232394
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -143.9574738, 137.5898285, -144.8092651, 131.6590424, -275.6165161, 281.7225342
1: -112.9016113, 130.6958466, -113.4549103, 124.2506790, -237.1522827, 242.5797119
2: -163.7802124, 143.6897125, -164.1941833, 138.0722961, -301.8525085, 306.0038147
3: -70.0736237, 166.9585724, -67.5839310, 165.3264618, -233.4506531, 234.5424957
4: -182.9930725, 142.4557343, -182.9160309, 137.3219604, -320.3150024, 324.6633911

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5090837, upper bound: 187.3970753
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5071755, upper bound: 187.6260446
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -210.2807465, 183.0075378, -88.6077271, 92.8103714, -303.0911255, 271.6152649
1: -165.1631775, 173.1105804, -69.5185242, 87.8645935, -253.0277405, 242.6291046
2: -238.5625763, 190.9392853, -100.9003143, 97.3328018, -335.8953552, 291.8395996
3: -94.2064362, 236.2539673, -47.0367928, 106.5991135, -200.0971680, 283.2907410
4: -265.5162659, 190.8201752, -113.0341187, 96.2283783, -361.7446289, 303.8543091

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5163652, upper bound: 187.4942331
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5117112, upper bound: 187.6259722
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -210.2807465, 183.0075378, -144.8092651, 131.6590424, -341.9397888, 327.8168030
1: -165.1631775, 173.1105804, -113.4549103, 124.2506790, -289.4138489, 286.5654907
2: -238.5625763, 190.9392853, -164.1941833, 138.0722961, -376.6348877, 355.1334839
3: -94.2064362, 236.2539673, -67.5839310, 165.3264618, -258.7338867, 303.8378296
4: -265.5162659, 190.8201752, -182.9160309, 137.3219604, -402.8381958, 373.7362061

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5163653, upper bound: 187.4969827
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5117113, upper bound: 187.6265915
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -143.9574738, 137.5898285, -143.9574738, 137.5898285, -280.9462280, 280.9462280
1: -112.9016113, 130.6958466, -112.9016113, 130.6958466, -242.0519104, 242.0519257
2: -163.7802124, 143.6897125, -163.7802124, 143.6897125, -305.8948975, 305.8948975
3: -70.0736237, 166.9585724, -70.0736237, 166.9585724, -235.3055420, 235.3055267
4: -182.9930725, 142.4557343, -182.9930725, 142.4557343, -324.9454956, 324.9454651

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6282766, upper bound: 187.3961512
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263684, upper bound: 187.6233484
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -143.9574738, 137.5898285, -210.2807465, 183.0075378, -326.9650269, 347.5987854
1: -112.9016113, 130.6958466, -165.1631775, 173.1105804, -286.0121765, 294.5191040
2: -163.7802124, 143.6897125, -238.5625763, 190.9392853, -354.7194824, 380.6860962
3: -70.0736237, 166.9585724, -94.2064362, 236.2539673, -304.4873352, 260.5887146
4: -182.9930725, 142.4557343, -265.5162659, 190.8201752, -373.8132324, 407.7260132

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6282766, upper bound: 187.4003479
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263684, upper bound: 187.6260440
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -210.2807465, 183.0075378, -143.9574738, 137.5898285, -347.5988159, 326.9650269
1: -165.1631775, 173.1105804, -112.9016113, 130.6958466, -294.5191040, 286.0121460
2: -238.5625763, 190.9392853, -163.7802124, 143.6897125, -380.6861572, 354.7194824
3: -94.2064362, 236.2539673, -70.0736237, 166.9585724, -260.5887146, 304.4873657
4: -265.5162659, 190.8201752, -182.9930725, 142.4557343, -407.7260132, 373.8132324

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6355582, upper bound: 187.4966932
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6309041, upper bound: 187.6261164
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -210.2807465, 183.0075378, -210.2807465, 183.0075378, -393.2882690, 393.2882690
1: -165.1631775, 173.1105804, -165.1631775, 173.1105804, -338.2737427, 338.2737427
2: -238.5625763, 190.9392853, -238.5625763, 190.9392853, -429.5018616, 429.5018311
3: -94.2064362, 236.2539673, -94.2064362, 236.2539673, -329.7705078, 329.7705383
4: -265.5162659, 190.8201752, -265.5162659, 190.8201752, -456.3364258, 456.3364258

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6355582, upper bound: 187.5017532
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6309042, upper bound: 187.6266972
time: 0.80 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.63 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6250098, upper bound: 187.3877502
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6240025, upper bound: 187.8149799
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6250098, upper bound: 187.3879327
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6240025, upper bound: 187.8151624
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6250098, upper bound: 187.3720326
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6241850, upper bound: 187.6240023
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6250098, upper bound: 187.3720998
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6241850, upper bound: 187.6240023
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6278511, upper bound: 187.3900510
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6268437, upper bound: 187.8172807
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6278511, upper bound: 187.3900510
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6268437, upper bound: 187.8172807
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6280475, upper bound: 187.3743333
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6270262, upper bound: 187.6263031
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6280475, upper bound: 187.3743334
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6270262, upper bound: 187.6263031
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6274424, upper bound: 187.4879346
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6263038, upper bound: 187.8149799
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6274424, upper bound: 187.4881171
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6263038, upper bound: 187.8151624
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6274664, upper bound: 187.4329129
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6263032, upper bound: 187.6268437
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6274664, upper bound: 187.4329129
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6263032, upper bound: 187.6268437
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6302836, upper bound: 187.4900308
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6291451, upper bound: 187.8149336
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6302836, upper bound: 187.4900804
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6291451, upper bound: 187.8150484
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6303077, upper bound: 187.4349487
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6291445, upper bound: 187.6273117
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6303077, upper bound: 187.4349487
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6291445, upper bound: 187.6273117
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5055578, upper bound: 187.3872464
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5045504, upper bound: 187.8144761
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5055578, upper bound: 187.3900602
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5045504, upper bound: 187.8172899
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5057542, upper bound: 187.3715288
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5047329, upper bound: 187.6234985
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5057542, upper bound: 187.3743426
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5047329, upper bound: 187.6263123
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6247507, upper bound: 187.3897308
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6237433, upper bound: 187.8169605
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6247507, upper bound: 187.3949111
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6237433, upper bound: 187.8221408
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5055578, upper bound: 187.3718545
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5045504, upper bound: 187.6234985
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6249471, upper bound: 187.3745005
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5045504, upper bound: 187.6234985
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5079903, upper bound: 187.4874308
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5068518, upper bound: 187.8144761
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5079903, upper bound: 187.4899905
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5068518, upper bound: 187.8144761
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5080144, upper bound: 187.4324091
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5068512, upper bound: 187.6263398
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5080144, upper bound: 187.4324091
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5068512, upper bound: 187.6269514
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6271832, upper bound: 187.4899152
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6260447, upper bound: 187.8169605
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6271832, upper bound: 187.4946240
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6260447, upper bound: 187.8169605
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6272073, upper bound: 187.4324091
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6260054, upper bound: 187.6263398
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6272073, upper bound: 187.4348991
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6260055, upper bound: 187.6269514
NS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6241992, upper bound: 187.3640684
NS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6234987, upper bound: 187.5045501
NS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6241992, upper bound: 187.3642509
NS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6234987, upper bound: 187.5047325
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6274597, upper bound: 187.4657145
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6263125, upper bound: 187.6257306
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6274597, upper bound: 187.4658970
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6263125, upper bound: 187.6259131
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6270404, upper bound: 187.3663691
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6263399, upper bound: 187.5068508
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6270404, upper bound: 187.3663691
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6263399, upper bound: 187.5068508
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6303010, upper bound: 187.4680153
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6291537, upper bound: 187.6263070
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6303010, upper bound: 187.4680153
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6291537, upper bound: 187.6264775
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6285358, upper bound: 187.3947653
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6266275, upper bound: 187.6237432
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6285358, upper bound: 187.3949487
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6266275, upper bound: 187.6237582
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6358173, upper bound: 187.4947343
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6311633, upper bound: 187.6264760
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6358173, upper bound: 187.4949191
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6311633, upper bound: 187.6264993
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6313771, upper bound: 187.3970661
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6294688, upper bound: 187.6260440
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6313771, upper bound: 187.3970662
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6294688, upper bound: 187.6260439
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6386586, upper bound: 187.4969826
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6340045, upper bound: 187.6269505
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6386586, upper bound: 187.4969978
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6340046, upper bound: 187.6269785
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5080076, upper bound: 187.4652107
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5068604, upper bound: 187.6252268
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5080077, upper bound: 187.4680245
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5068604, upper bound: 187.6259756
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6239400, upper bound: 187.3641368
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6232395, upper bound: 187.5046185
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6239400, upper bound: 187.3663691
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6232395, upper bound: 187.5068508
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6272006, upper bound: 187.4659287
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6260533, upper bound: 187.6257991
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6272006, upper bound: 187.4684190
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6260533, upper bound: 187.6264775
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5090837, upper bound: 187.3942621
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5071755, upper bound: 187.6232394
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5090837, upper bound: 187.3970753
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5071755, upper bound: 187.6260446
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5163652, upper bound: 187.4942331
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5117112, upper bound: 187.6259722
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5163653, upper bound: 187.4969827
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.5117113, upper bound: 187.6265915
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6282766, upper bound: 187.3961512
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6263684, upper bound: 187.6233484
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6282766, upper bound: 187.4003479
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6263684, upper bound: 187.6260440
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6355582, upper bound: 187.4966932
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6309041, upper bound: 187.6261164
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6355582, upper bound: 187.5017532
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 3, lower bound: -187.6309042, upper bound: 187.6266972

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -34.2662277, 50.8938408, -40.9809875, 54.7791138, -89.0453415, 91.8748169
1: -27.0025425, 47.7687073, -32.1188278, 51.3675804, -78.3701172, 79.8875198
2: -39.5765114, 53.9266510, -47.0364113, 57.8066292, -97.3831406, 100.9630585
3: -25.4885406, 48.3789978, -27.2729225, 55.0080185, -80.4965515, 75.6519165
4: -44.9341927, 53.1512604, -53.0424118, 57.1621284, -102.0963211, 106.1936646

Time for backsubstitution: 1.93 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.61 + 417.60 = 421.21 seconds
