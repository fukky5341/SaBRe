## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 5)
Time budget: 1800 seconds
Split limit: 100
Threshold: 27.5662213848


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0799713, 51.0799675)
1: (-19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778)
2: (-13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6210327, 29.6210327)
3: (-14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0640717, 37.0640640)
4: (-18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209)
5: (-16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765)
6: (-25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520)
7: (-23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790)
8: (-20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.4103546, 44.4103470)
9: (-14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724)
10: (-29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808)
11: (-33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669)
12: (-27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4790344, 39.4790382)
13: (-18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208)
14: (-56.6111145, -1.5055046, -56.6111145, -1.5055046, -50.0486603, 50.0486603)
15: (-21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448)
16: (-33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847)
17: (-62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.1339264, 62.1339340)
18: (-34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.9741211, 36.9741211)
19: (-27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667)
20: (-19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7679443, 28.7679482)
21: (-31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973)
22: (-32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4571991, 38.4571953)
23: (-23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009)
24: (-28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526)
25: (-22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5885162, 33.5885124)
26: (-34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.8358078, 43.8358040)
27: (-28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267)
28: (-22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785)
29: (-34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828)
30: (-25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035)
31: (-34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786)
32: (-20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219)
33: (-30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1827240, 51.1827164)
34: (-28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951)
35: (-25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409)
36: (-24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5372620, 43.5372620)
37: (-44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2858734, 58.2858810)
38: (-33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559)
39: (-34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3795395, 51.3795471)
40: (-34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.7035675, 49.7035675)
41: (-24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897)
42: (-16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.27 + 103.83 = 106.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 13, lower bound: -27.5938152, upper bound: 27.5938152

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1537

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1689

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5925953, upper bound: 27.5905191
time: 38.82 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5905191, upper bound: 27.5925953
time: 55.41 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 94.35 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 94.35
Output dim: 13, lower bound: -27.5925953, upper bound: 27.5905191
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 94.35
Output dim: 13, lower bound: -27.5905191, upper bound: 27.5925953

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0796204, 51.0797729
1: -19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778
2: -13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6208496, 29.6208687
3: -14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0605927, 37.0620422
4: -18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209
5: -16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765
6: -25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520
7: -23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790
8: -20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.4091949, 44.4095840
9: -14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724
10: -29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808
11: -33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669
12: -27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4756088, 39.4728546
13: -18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208
14: -56.6111145, -1.5055046, -56.6111145, -1.5055046, -50.0395203, 50.0321808
15: -21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448
16: -33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847
17: -62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.1296387, 62.1261902
18: -34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.9712296, 36.9682465
19: -27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667
20: -19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7661133, 28.7653427
21: -31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973
22: -32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4562225, 38.4554214
23: -23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009
24: -28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526
25: -22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5875854, 33.5872803
26: -34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.8331375, 43.8295364
27: -28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267
28: -22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785
29: -34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828
30: -25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035
31: -34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786
32: -20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219
33: -30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1782684, 51.1802444
34: -28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951
35: -25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409
36: -24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5366135, 43.5369949
37: -44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2815552, 58.2837143
38: -33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559
39: -34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3785324, 51.3789444
40: -34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.7017670, 49.7025604
41: -24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897
42: -16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1537

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5920092, upper bound: 27.5525190
time: 53.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5546001, upper bound: 27.5899333
time: 40.86 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0797577, 51.0796204
1: -19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778
2: -13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6208649, 29.6208458
3: -14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0620499, 37.0605888
4: -18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209
5: -16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765
6: -25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520
7: -23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790
8: -20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.4095917, 44.4091949
9: -14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724
10: -29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808
11: -33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669
12: -27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4728546, 39.4756088
13: -18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208
14: -56.6111145, -1.5055046, -56.6111145, -1.5055046, -50.0321655, 50.0395126
15: -21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448
16: -33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847
17: -62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.1261902, 62.1296387
18: -34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.9682465, 36.9712296
19: -27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667
20: -19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7653427, 28.7661171
21: -31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973
22: -32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4554214, 38.4562225
23: -23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009
24: -28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526
25: -22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5872803, 33.5875854
26: -34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.8295364, 43.8331375
27: -28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267
28: -22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785
29: -34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828
30: -25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035
31: -34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786
32: -20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219
33: -30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1802521, 51.1782608
34: -28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951
35: -25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409
36: -24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5369949, 43.5366135
37: -44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2837067, 58.2815552
38: -33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559
39: -34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3789291, 51.3785477
40: -34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.7025604, 49.7017517
41: -24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897
42: -16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1537

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 701

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5899333, upper bound: 27.5546001
time: 76.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5525190, upper bound: 27.5920092
time: 188.68 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 267.59 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 267.59
Output dim: 13, lower bound: -27.5920092, upper bound: 27.5525190
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 267.59
Output dim: 13, lower bound: -27.5546001, upper bound: 27.5899333
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 267.59
Output dim: 13, lower bound: -27.5899333, upper bound: 27.5546001
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 267.59
Output dim: 13, lower bound: -27.5525190, upper bound: 27.5920092

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0781097, 51.0780792
1: -19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778
2: -13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6165047, 29.6168213
3: -14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0535965, 37.0537415
4: -18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209
5: -16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765
6: -25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520
7: -23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790
8: -20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.4054260, 44.4052048
9: -14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724
10: -29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808
11: -33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669
12: -27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4786530, 39.4778061
13: -18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208
14: -56.6111145, -1.5055046, -56.6111145, -1.5055046, -50.0510864, 50.0473595
15: -21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448
16: -33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847
17: -62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.1365509, 62.1344147
18: -34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.9530563, 36.9529724
19: -27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667
20: -19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7579384, 28.7583389
21: -31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973
22: -32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4507904, 38.4508629
23: -23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009
24: -28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526
25: -22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5828247, 33.5830078
26: -34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.8163681, 43.8153419
27: -28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267
28: -22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785
29: -34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828
30: -25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035
31: -34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786
32: -20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219
33: -30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1746292, 51.1754379
34: -28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951
35: -25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409
36: -24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5365982, 43.5369492
37: -44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2797699, 58.2811050
38: -33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559
39: -34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3786469, 51.3790131
40: -34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.7017670, 49.7025604
41: -24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897
42: -16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1537

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 573

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5266058, upper bound: 27.5520030
time: 69.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5914911, upper bound: 27.5245195
time: 40.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0779266, 51.0782700
1: -19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778
2: -13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6168022, 29.6165237
3: -14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0522919, 37.0550537
4: -18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209
5: -16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765
6: -25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520
7: -23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790
8: -20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.4048157, 44.4058151
9: -14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724
10: -29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808
11: -33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669
12: -27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4805603, 39.4759064
13: -18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208
14: -56.6111145, -1.5055046, -56.6111145, -1.5055046, -50.0546875, 50.0437698
15: -21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448
16: -33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847
17: -62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.1378632, 62.1331100
18: -34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.9559555, 36.9500771
19: -27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667
20: -19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7591057, 28.7571716
21: -31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973
22: -32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4516602, 38.4499893
23: -23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009
24: -28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526
25: -22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5833206, 33.5825157
26: -34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.8189468, 43.8127632
27: -28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267
28: -22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785
29: -34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828
30: -25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035
31: -34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786
32: -20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219
33: -30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1734390, 51.1766281
34: -28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951
35: -25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409
36: -24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5365677, 43.5369720
37: -44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2789459, 58.2819290
38: -33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559
39: -34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3786163, 51.3790512
40: -34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.7017517, 49.7025757
41: -24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897
42: -16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1537

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 573

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5266058, upper bound: 27.5894153
time: 57.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5540836, upper bound: 27.5619449
time: 40.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0782623, 51.0779266
1: -19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778
2: -13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6165276, 29.6167984
3: -14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0550537, 37.0522881
4: -18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209
5: -16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765
6: -25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520
7: -23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790
8: -20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.4058075, 44.4048080
9: -14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724
10: -29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808
11: -33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669
12: -27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4759064, 39.4805603
13: -18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208
14: -56.6111145, -1.5055046, -56.6111145, -1.5055046, -50.0437622, 50.0546913
15: -21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448
16: -33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847
17: -62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.1331024, 62.1378632
18: -34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.9500809, 36.9559555
19: -27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667
20: -19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7571754, 28.7591095
21: -31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973
22: -32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4499893, 38.4516602
23: -23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009
24: -28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526
25: -22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5825195, 33.5833130
26: -34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.8127670, 43.8189392
27: -28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267
28: -22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785
29: -34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828
30: -25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035
31: -34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786
32: -20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219
33: -30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1766281, 51.1734543
34: -28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951
35: -25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409
36: -24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5369797, 43.5365753
37: -44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2819214, 58.2789459
38: -33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559
39: -34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3790436, 51.3786163
40: -34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.7025757, 49.7017517
41: -24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897
42: -16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1537

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 573

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5619448, upper bound: 27.5540837
time: 53.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5894153, upper bound: 27.5266058
time: 54.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0780792, 51.0781174
1: -19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778
2: -13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6168251, 29.6165009
3: -14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0537415, 37.0535965
4: -18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209
5: -16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765
6: -25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520
7: -23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790
8: -20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.4052124, 44.4054184
9: -14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724
10: -29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808
11: -33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669
12: -27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4778137, 39.4786568
13: -18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208
14: -56.6111145, -1.5055046, -56.6111145, -1.5055046, -50.0473633, 50.0511017
15: -21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448
16: -33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847
17: -62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.1344147, 62.1365509
18: -34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.9529648, 36.9530640
19: -27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667
20: -19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7583351, 28.7579460
21: -31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973
22: -32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4508667, 38.4507904
23: -23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009
24: -28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526
25: -22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5830154, 33.5828209
26: -34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.8153458, 43.8163643
27: -28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267
28: -22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785
29: -34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828
30: -25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035
31: -34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786
32: -20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219
33: -30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1754379, 51.1746445
34: -28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951
35: -25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409
36: -24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5369492, 43.5365982
37: -44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2810974, 58.2797699
38: -33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559
39: -34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3790131, 51.3786545
40: -34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.7025604, 49.7017670
41: -24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897
42: -16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1537

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 573

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5245195, upper bound: 27.5914911
time: 62.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5520030, upper bound: 27.5640276
time: 138.27 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 203.21 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 203.21
Output dim: 13, lower bound: -27.5266058, upper bound: 27.5520030
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 203.21
Output dim: 13, lower bound: -27.5914911, upper bound: 27.5245195
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 203.21
Output dim: 13, lower bound: -27.5266058, upper bound: 27.5894153
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 203.21
Output dim: 13, lower bound: -27.5540836, upper bound: 27.5619449
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 203.21
Output dim: 13, lower bound: -27.5619448, upper bound: 27.5540837
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 203.21
Output dim: 13, lower bound: -27.5894153, upper bound: 27.5266058
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 203.21
Output dim: 13, lower bound: -27.5245195, upper bound: 27.5914911
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 203.21
Output dim: 13, lower bound: -27.5520030, upper bound: 27.5640276

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0805435, 51.0786133
1: -19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778
2: -13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6128693, 29.6116753
3: -14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0578461, 37.0537720
4: -18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209
5: -16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765
6: -25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520
7: -23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790
8: -20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.4102783, 44.4080811
9: -14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724
10: -29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808
11: -33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669
12: -27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4614716, 39.4672546
13: -18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208
14: -56.6111145, -1.5055046, -56.6111145, -1.5055046, -49.9643097, 49.9747696
15: -21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448
16: -33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847
17: -62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.1013336, 62.1051331
18: -34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.9008522, 36.9098091
19: -27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667
20: -19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7509384, 28.7550697
21: -31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973
22: -32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4347610, 38.4374542
23: -23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009
24: -28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526
25: -22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5733719, 33.5756187
26: -34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.7647552, 43.7726326
27: -28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267
28: -22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785
29: -34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828
30: -25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035
31: -34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786
32: -20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219
33: -30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1525650, 51.1490479
34: -28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951
35: -25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409
36: -24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5347061, 43.5346756
37: -44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2632370, 58.2608643
38: -33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559
39: -34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3767166, 51.3766937
40: -34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.6981659, 49.6981659
41: -24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897
42: -16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1537

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 700

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5910457, upper bound: 27.5120159
time: 46.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5708914, upper bound: 27.5233185
time: 56.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0784683, 51.0806885
1: -19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778
2: -13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6116486, 29.6128960
3: -14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0523148, 37.0592995
4: -18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209
5: -16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765
6: -25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520
7: -23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790
8: -20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.4076843, 44.4106750
9: -14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724
10: -29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808
11: -33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669
12: -27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4700012, 39.4587135
13: -18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208
14: -56.6111145, -1.5055046, -56.6111145, -1.5055046, -49.9821014, 49.9569855
15: -21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448
16: -33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847
17: -62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.1085815, 62.0978851
18: -34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.9127922, 36.8978691
19: -27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667
20: -19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7558441, 28.7501717
21: -31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973
22: -32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4382553, 38.4339638
23: -23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009
24: -28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526
25: -22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5759277, 33.5730629
26: -34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.7762299, 43.7611618
27: -28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267
28: -22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785
29: -34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828
30: -25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035
31: -34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786
32: -20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219
33: -30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1470566, 51.1545639
34: -28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951
35: -25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409
36: -24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5343018, 43.5350876
37: -44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2586899, 58.2654114
38: -33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559
39: -34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3763046, 51.3771095
40: -34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.6973724, 49.6989670
41: -24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897
42: -16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1537

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 700

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5254024, upper bound: 27.5688393
time: 54.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5140930, upper bound: 27.5889730
time: 76.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0806808, 51.0784645
1: -19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778
2: -13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6128998, 29.6116524
3: -14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0593033, 37.0523148
4: -18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209
5: -16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765
6: -25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520
7: -23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790
8: -20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.4106750, 44.4076843
9: -14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724
10: -29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808
11: -33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669
12: -27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4587097, 39.4700050
13: -18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208
14: -56.6111145, -1.5055046, -56.6111145, -1.5055046, -49.9569855, 49.9821014
15: -21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448
16: -33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847
17: -62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.0978851, 62.1085815
18: -34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.8978691, 36.9127922
19: -27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667
20: -19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7501678, 28.7558441
21: -31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973
22: -32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4339600, 38.4382553
23: -23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009
24: -28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526
25: -22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5730667, 33.5759239
26: -34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.7611618, 43.7762299
27: -28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267
28: -22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785
29: -34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828
30: -25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035
31: -34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786
32: -20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219
33: -30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1545486, 51.1470566
34: -28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951
35: -25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409
36: -24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5350876, 43.5343018
37: -44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2654037, 58.2587051
38: -33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559
39: -34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3771133, 51.3762970
40: -34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.6989594, 49.6973648
41: -24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897
42: -16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1537

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 700

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5889729, upper bound: 27.5140930
time: 39.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5688393, upper bound: 27.5254024
time: 57.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0786057, 51.0805397
1: -19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778
2: -13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6116791, 29.6128731
3: -14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0537720, 37.0578461
4: -18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209
5: -16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765
6: -25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520
7: -23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790
8: -20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.4080811, 44.4102783
9: -14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724
10: -29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808
11: -33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669
12: -27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4672546, 39.4614677
13: -18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208
14: -56.6111145, -1.5055046, -56.6111145, -1.5055046, -49.9747620, 49.9643173
15: -21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448
16: -33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847
17: -62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.1051331, 62.1013336
18: -34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.9098091, 36.9008560
19: -27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667
20: -19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7550659, 28.7509422
21: -31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973
22: -32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4374542, 38.4347610
23: -23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009
24: -28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526
25: -22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5756226, 33.5733681
26: -34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.7726288, 43.7647552
27: -28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267
28: -22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785
29: -34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828
30: -25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035
31: -34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786
32: -20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219
33: -30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1490555, 51.1525726
34: -28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951
35: -25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409
36: -24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5346756, 43.5347061
37: -44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2608566, 58.2632523
38: -33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559
39: -34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3767014, 51.3767128
40: -34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.6981659, 49.6981583
41: -24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897
42: -16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1537

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 700

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5233185, upper bound: 27.5708914
time: 57.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5120159, upper bound: 27.5910457
time: 50.04 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 109.34 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 109.34
Output dim: 13, lower bound: -27.5910457, upper bound: 27.5120159
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 109.34
Output dim: 13, lower bound: -27.5708914, upper bound: 27.5233185
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 109.34
Output dim: 13, lower bound: -27.5254024, upper bound: 27.5688393
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 109.34
Output dim: 13, lower bound: -27.5140930, upper bound: 27.5889730
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 109.34
Output dim: 13, lower bound: -27.5889729, upper bound: 27.5140930
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 109.34
Output dim: 13, lower bound: -27.5688393, upper bound: 27.5254024
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 109.34
Output dim: 13, lower bound: -27.5233185, upper bound: 27.5708914
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 109.34
Output dim: 13, lower bound: -27.5120159, upper bound: 27.5910457

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0830383, 51.0813866
1: -19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778
2: -13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6119232, 29.6110992
3: -14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0532608, 37.0481834
4: -18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209
5: -16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765
6: -25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520
7: -23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790
8: -20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.4083557, 44.4059906
9: -14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724
10: -29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808
11: -33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669
12: -27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4708405, 39.4782867
13: -18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208
14: -56.6111145, -1.5055046, -56.6111145, -1.5055046, -49.9505157, 49.9651260
15: -21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448
16: -33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847
17: -62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.1075287, 62.1130676
18: -34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.8844490, 36.8955154
19: -27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667
20: -19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7447243, 28.7492714
21: -31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973
22: -32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4308701, 38.4341125
23: -23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009
24: -28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526
25: -22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5696220, 33.5718269
26: -34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.7621384, 43.7722359
27: -28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267
28: -22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785
29: -34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828
30: -25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035
31: -34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786
32: -20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219
33: -30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1458740, 51.1412125
34: -28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951
35: -25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409
36: -24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5342712, 43.5341415
37: -44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2577209, 58.2542496
38: -33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559
39: -34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3762207, 51.3761139
40: -34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.6971130, 49.6969070
41: -24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897
42: -16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1537

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5906008, upper bound: 27.4897349
time: 39.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5693407, upper bound: 27.5115701
time: 61.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0832977, 51.0810051
1: -19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778
2: -13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6123047, 29.6106758
3: -14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0522614, 37.0491867
4: -18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209
5: -16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765
6: -25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520
7: -23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790
8: -20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.4081879, 44.4060822
9: -14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724
10: -29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808
11: -33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669
12: -27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4725037, 39.4766273
13: -18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208
14: -56.6111145, -1.5055046, -56.6111145, -1.5055046, -49.9546661, 49.9609833
15: -21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448
16: -33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847
17: -62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.1092682, 62.1113358
18: -34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.8865623, 36.8934059
19: -27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667
20: -19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7450294, 28.7488518
21: -31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973
22: -32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4314270, 38.4335632
23: -23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009
24: -28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526
25: -22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5694618, 33.5718651
26: -34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.7643509, 43.7700119
27: -28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267
28: -22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785
29: -34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828
30: -25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035
31: -34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786
32: -20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219
33: -30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1447296, 51.1423492
34: -28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951
35: -25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409
36: -24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5341644, 43.5342407
37: -44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2566528, 58.2553406
38: -33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559
39: -34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3761292, 51.3762093
40: -34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.6968994, 49.6971207
41: -24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897
42: -16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1537

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5704481, upper bound: 27.5013504
time: 43.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5467234, upper bound: 27.5228699
time: 44.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -37.0122108, 14.1888361, -37.0122108, 14.1888361, -51.0808563, 51.0834618
1: -19.7674370, 16.4605408, -19.7674370, 16.4605408, -36.2279778, 36.2279778
2: -13.6377487, 16.6178360, -13.6377487, 16.6178360, -29.6106567, 29.6123199
3: -14.0041981, 23.4930649, -14.0041981, 23.4930649, -37.0477295, 37.0537148
4: -18.6668510, 18.1617718, -18.6668510, 18.1617718, -36.8286209, 36.8286209
5: -16.1674290, 20.0294495, -16.1674290, 20.0294495, -36.1968765, 36.1968765
6: -25.9964066, 14.0117455, -25.9964066, 14.0117455, -40.0081520, 40.0081520
7: -23.3628426, 18.8906364, -23.3628426, 18.8906364, -42.2534790, 42.2534790
8: -20.7247581, 23.7543182, -20.7247581, 23.7543182, -44.4057007, 44.4085846
9: -14.7718945, 19.4907761, -14.7718945, 19.4907761, -34.2626724, 34.2626724
10: -29.7552662, 17.2172127, -29.7552662, 17.2172127, -46.9724808, 46.9724808
11: -33.8062286, 7.4790382, -33.8062286, 7.4790382, -41.2852669, 41.2852669
12: -27.9611130, 11.9503508, -27.9611130, 11.9503508, -39.4793777, 39.4697495
13: -18.1574726, 28.4990482, -18.1574726, 28.4990482, -46.6565208, 46.6565208
14: -56.6111145, -1.5055046, -56.6111145, -1.5055046, -49.9683075, 49.9473419
15: -21.8218689, 17.5952778, -21.8218689, 17.5952778, -39.4171448, 39.4171448
16: -33.0908661, 13.7911186, -33.0908661, 13.7911186, -46.8819847, 46.8819847
17: -62.9188080, 0.6972713, -62.9188080, 0.6972713, -62.1147614, 62.1058121
18: -34.8533096, 3.7487707, -34.8533096, 3.7487707, -36.8963890, 36.8835754
19: -27.3268147, 3.1837530, -27.3268147, 3.1837530, -30.5105667, 30.5105667
20: -19.2003727, 10.2061882, -19.2003727, 10.2061882, -28.7496223, 28.7442589
21: -31.7780838, 4.4144154, -31.7780838, 4.4144154, -36.1924973, 36.1924973
22: -32.2084808, 6.5936913, -32.2084808, 6.5936913, -38.4343567, 38.4306297
23: -23.4332409, 7.5414596, -23.4332409, 7.5414596, -30.9747009, 30.9747009
24: -28.0918465, 9.4708061, -28.0918465, 9.4708061, -37.5626526, 37.5626526
25: -22.0050449, 11.6457996, -22.0050449, 11.6457996, -33.5721703, 33.5691566
26: -34.9116592, 10.7799397, -34.9116592, 10.7799397, -43.7736130, 43.7607574
27: -28.7818832, 7.5861425, -28.7818832, 7.5861425, -36.3680267, 36.3680267
28: -22.4773407, 12.6479378, -22.4773407, 12.6479378, -35.1252785, 35.1252785
29: -34.4301186, 3.9525642, -34.4301186, 3.9525642, -38.3826828, 38.3826828
30: -25.9085217, 12.2365799, -25.9085217, 12.2365799, -38.1451035, 38.1451035
31: -34.2741089, 6.6356697, -34.2741089, 6.6356697, -40.9097786, 40.9097786
32: -20.7110062, 13.4547157, -20.7110062, 13.4547157, -34.1657219, 34.1657219
33: -30.1621819, 21.1929989, -30.1621819, 21.1929989, -51.1403656, 51.1467285
34: -28.8355064, 17.1622887, -28.8355064, 17.1622887, -45.9977951, 45.9977951
35: -25.9519234, 20.3029175, -25.9519234, 20.3029175, -46.2548409, 46.2548409
36: -24.5922203, 18.9871521, -24.5922203, 18.9871521, -43.5338593, 43.5345459
37: -44.7283478, 13.7959728, -44.7283478, 13.7959728, -58.2531738, 58.2588043
38: -33.0869408, 18.3433151, -33.0869408, 18.3433151, -51.4302559, 51.4302559
39: -34.7091904, 16.8242455, -34.7091904, 16.8242455, -51.3758087, 51.3765259
40: -34.6185150, 15.5769863, -34.6185150, 15.5769863, -49.6963043, 49.6977081
41: -24.5610046, 14.6855869, -24.5610046, 14.6855869, -39.2465897, 39.2465897
42: -16.4853477, 11.1011610, -16.4853477, 11.1011610, -27.5865097, 27.5865097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=119, inp2_unstable=119, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1537

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5249548, upper bound: 27.5467234
time: 60.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.4918077, upper bound: 27.5683969
time: 40.99 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 103.34 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 103.34
Output dim: 13, lower bound: -27.5906008, upper bound: 27.4897349
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 103.34
Output dim: 13, lower bound: -27.5693407, upper bound: 27.5115701
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 103.34
Output dim: 13, lower bound: -27.5704481, upper bound: 27.5013504
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 103.34
Output dim: 13, lower bound: -27.5467234, upper bound: 27.5228699
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 103.34
Output dim: 13, lower bound: -27.5249548, upper bound: 27.5467234
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 103.34
Output dim: 13, lower bound: -27.4918077, upper bound: 27.5683969
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 103.34
Output dim: 13, lower bound: -27.5140930, upper bound: 27.5889730
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 103.34
Output dim: 13, lower bound: -27.5889729, upper bound: 27.5140930
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 103.34
Output dim: 13, lower bound: -27.5688393, upper bound: 27.5254024
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 103.34
Output dim: 13, lower bound: -27.5233185, upper bound: 27.5708914
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 103.34
Output dim: 13, lower bound: -27.5120159, upper bound: 27.5910457

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 106.10 + 1725.35 = 1831.46 seconds
