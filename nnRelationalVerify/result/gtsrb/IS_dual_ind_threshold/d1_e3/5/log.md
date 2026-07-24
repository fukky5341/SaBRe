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
execution time: IAR + RelationalAnalysis = 2.32 + 104.60 = 106.92 seconds
status: Status.UNKNOWN
relational distance
Output dim: 13, lower bound: -27.5938152, upper bound: 27.5938152

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 753

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 605

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5428236, upper bound: 27.5854763
time: 38.57 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5428236, upper bound: 27.5925058
time: 56.11 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 94.80 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 94.80
Output dim: 13, lower bound: -27.5428236, upper bound: 27.5854763
IS_A2, status: Status.UNKNOWN, split count: 1, time: 94.80
Output dim: 13, lower bound: -27.5428236, upper bound: 27.5925058

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -36.9724350, 14.1625175, -36.9905167, 14.1754704, -51.0138855, 51.0217743
1: -19.7439575, 16.4470978, -19.7563934, 16.4545822, -36.1985397, 36.2034912
2: -13.5746861, 16.5940285, -13.6082287, 16.6117554, -29.5481720, 29.5643005
3: -13.9017372, 23.4508362, -13.9556408, 23.4844971, -36.9500122, 36.9702950
4: -18.6024933, 18.1306305, -18.6371002, 18.1527348, -36.7552261, 36.7677307
5: -16.0666924, 19.9922409, -16.1197453, 20.0225430, -36.0892334, 36.1119843
6: -25.9582539, 13.9980726, -25.9792843, 14.0053883, -39.9636421, 39.9773560
7: -23.3123703, 18.8709106, -23.3387489, 18.8858948, -42.1982651, 42.2096596
8: -20.6557007, 23.7192726, -20.6922455, 23.7433090, -44.3271027, 44.3421173
9: -14.7358189, 19.4574890, -14.7539091, 19.4753036, -34.2111206, 34.2113991
10: -29.7207317, 17.1267281, -29.7449112, 17.1747608, -46.8954926, 46.8716393
11: -33.7711716, 7.3890004, -33.7969551, 7.4370952, -41.2082672, 41.1859550
12: -27.9264946, 11.8673573, -27.9515915, 11.9116783, -39.4093246, 39.3815536
13: -18.0067749, 28.4605522, -18.0854950, 28.4915199, -46.4982948, 46.5460472
14: -56.5649986, -1.6014709, -56.5992432, -1.5506096, -49.9564819, 49.9373550
15: -21.7800102, 17.5726776, -21.8011703, 17.5853138, -39.3653259, 39.3738480
16: -33.0484505, 13.7200222, -33.0680275, 13.7556705, -46.8041229, 46.7880478
17: -62.9028549, 0.6584644, -62.9091492, 0.6800709, -62.1042633, 62.0838318
18: -34.8255424, 3.6379547, -34.8481941, 3.6956158, -36.8860855, 36.8498917
19: -27.2919197, 3.1198144, -27.3179741, 3.1529441, -30.4448643, 30.4377880
20: -19.1839790, 10.1539431, -19.1946507, 10.1817770, -28.7206955, 28.7065392
21: -31.7404861, 4.3496981, -31.7675209, 4.3833323, -36.1238174, 36.1172180
22: -32.1753311, 6.5296679, -32.1993561, 6.5630732, -38.3919525, 38.3824768
23: -23.3968925, 7.4482446, -23.4253254, 7.4973197, -30.8942127, 30.8735695
24: -28.0525131, 9.3722496, -28.0837555, 9.4237490, -37.4762611, 37.4560051
25: -21.9731503, 11.5699997, -21.9965343, 11.6094131, -33.5180244, 33.5022202
26: -34.8875275, 10.6712027, -34.9052544, 10.7291384, -43.7541656, 43.7171097
27: -28.7392025, 7.4830551, -28.7726192, 7.5368109, -36.2760124, 36.2556763
28: -22.4487114, 12.5685158, -22.4704990, 12.6102886, -35.0589981, 35.0390167
29: -34.3910942, 3.8877792, -34.4194946, 3.9217920, -38.3128853, 38.3072739
30: -25.8789825, 12.1681156, -25.8998795, 12.2043924, -38.0833740, 38.0679932
31: -34.2247543, 6.5468388, -34.2623901, 6.5926580, -40.8174133, 40.8092270
32: -20.6740246, 13.4111490, -20.6980648, 13.4333124, -34.1073380, 34.1092148
33: -30.0593948, 21.1905804, -30.1137161, 21.1844559, -51.0740814, 51.1332855
34: -28.8091469, 17.1063499, -28.8259888, 17.1355591, -45.9447060, 45.9323387
35: -25.8854485, 20.3002567, -25.9204292, 20.2970924, -46.1825409, 46.2206879
36: -24.5467625, 18.9636269, -24.5715485, 18.9759350, -43.4791794, 43.4916611
37: -44.6685524, 13.7372284, -44.7037888, 13.7683392, -58.1840973, 58.1979523
38: -33.0350151, 18.2825985, -33.0628967, 18.3149109, -51.3499260, 51.3454971
39: -34.6137238, 16.7903290, -34.6661911, 16.8074379, -51.2655182, 51.3011131
40: -34.5751228, 15.5468378, -34.5994415, 15.5623398, -49.6434174, 49.6522446
41: -24.5259972, 14.6367130, -24.5453529, 14.6616392, -39.1876373, 39.1820679
42: -16.4630890, 11.0691118, -16.4739246, 11.0862560, -27.5493450, 27.5430374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=119, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 753

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 637

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5408336, upper bound: 27.5562106
time: 44.40 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5408336, upper bound: 27.5844494
time: 39.45 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -37.0105858, 14.1829233, -37.0116959, 14.1869125, -51.0855713, 51.0751305
1: -19.7664852, 16.4576302, -19.7671356, 16.4596653, -36.2261505, 36.2247658
2: -13.6364088, 16.6171665, -13.6373224, 16.6176224, -29.6078033, 29.6199303
3: -14.0020943, 23.4920349, -14.0035601, 23.4927521, -37.0384064, 37.0622978
4: -18.6655102, 18.1607761, -18.6664238, 18.1614609, -36.8269730, 36.8272018
5: -16.1653442, 20.0286350, -16.1667824, 20.0292053, -36.1945496, 36.1954193
6: -25.9945183, 14.0101881, -25.9958191, 14.0112667, -40.0057831, 40.0060081
7: -23.3613338, 18.8897839, -23.3623619, 18.8903809, -42.2517166, 42.2521439
8: -20.7232780, 23.7528458, -20.7242870, 23.7538414, -44.3965759, 44.4083481
9: -14.7709475, 19.4890385, -14.7715979, 19.4902534, -34.2612000, 34.2606354
10: -29.7543888, 17.2151871, -29.7549858, 17.2166176, -46.9710083, 46.9701729
11: -33.8048744, 7.4773593, -33.8058167, 7.4785151, -41.2833900, 41.2831764
12: -27.9597721, 11.9486036, -27.9606915, 11.9498043, -39.4770355, 39.4479218
13: -18.1542969, 28.4979324, -18.1565018, 28.4986839, -46.6529808, 46.6544342
14: -56.6098022, -1.5072899, -56.6106949, -1.5060673, -50.0464935, 49.9839325
15: -21.8201561, 17.5942268, -21.8213348, 17.5949478, -39.4151039, 39.4155617
16: -33.0891914, 13.7892685, -33.0903435, 13.7905607, -46.8797531, 46.8796120
17: -62.9171829, 0.6933937, -62.9183273, 0.6960869, -62.1290436, 62.1574554
18: -34.8523445, 3.7466412, -34.8530121, 3.7481165, -36.9723282, 36.9208298
19: -27.3256950, 3.1825070, -27.3264694, 3.1833816, -30.5090771, 30.5089760
20: -19.1994591, 10.2050562, -19.2000866, 10.2058420, -28.7634926, 28.7443581
21: -31.7767296, 4.4131508, -31.7776604, 4.4140358, -36.1907654, 36.1908112
22: -32.2070351, 6.5923781, -32.2080154, 6.5932775, -38.4552917, 38.4403458
23: -23.4322891, 7.5397472, -23.4329491, 7.5409355, -30.9732246, 30.9726963
24: -28.0906601, 9.4689274, -28.0914726, 9.4702435, -37.5609055, 37.5604019
25: -22.0041046, 11.6442394, -22.0047455, 11.6453028, -33.5870743, 33.5737228
26: -34.9102364, 10.7778015, -34.9112091, 10.7792645, -43.8303299, 43.7856941
27: -28.7804623, 7.5843029, -28.7814331, 7.5855894, -36.3660507, 36.3657379
28: -22.4764614, 12.6464043, -22.4770603, 12.6474409, -35.1239014, 35.1234665
29: -34.4283638, 3.9513245, -34.4295883, 3.9521866, -38.3805504, 38.3809128
30: -25.9069519, 12.2351789, -25.9080524, 12.2361412, -38.1430931, 38.1432304
31: -34.2727547, 6.6338768, -34.2736740, 6.6351223, -40.9078751, 40.9073410
32: -20.7098808, 13.4536095, -20.7106590, 13.4543972, -34.1642761, 34.1642685
33: -30.1593227, 21.1922512, -30.1613159, 21.1927681, -51.1842422, 51.1808472
34: -28.8346634, 17.1609097, -28.8352509, 17.1618290, -45.9964905, 45.9961624
35: -25.9482536, 20.3023605, -25.9506836, 20.3027363, -46.2509918, 46.2530441
36: -24.5888958, 18.9859238, -24.5911980, 18.9867878, -43.5336304, 43.5350113
37: -44.7265472, 13.7947159, -44.7278214, 13.7955818, -58.2987061, 58.2823715
38: -33.0843201, 18.3418465, -33.0861320, 18.3428726, -51.4271927, 51.4279785
39: -34.7040253, 16.8232746, -34.7075768, 16.8239555, -51.3743210, 51.3769836
40: -34.6153183, 15.5761156, -34.6175308, 15.5767326, -49.7004852, 49.7016449
41: -24.5595856, 14.6840019, -24.5605850, 14.6851063, -39.2446899, 39.2445869
42: -16.4841156, 11.0993462, -16.4849663, 11.1005669, -27.5846825, 27.5843124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=119, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 753

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 637

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5408336, upper bound: 27.5636279
time: 44.12 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5913730, upper bound: 27.5913733
time: 40.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 86.42 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 86.42
Output dim: 13, lower bound: -27.5408336, upper bound: 27.5562106
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 86.42
Output dim: 13, lower bound: -27.5408336, upper bound: 27.5844494
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 86.42
Output dim: 13, lower bound: -27.5408336, upper bound: 27.5636279
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 86.42
Output dim: 13, lower bound: -27.5913730, upper bound: 27.5913733

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -36.9710083, 14.1613331, -36.9878311, 14.1731014, -51.0076218, 51.0100441
1: -19.7426395, 16.4463310, -19.7537289, 16.4530411, -36.1956787, 36.2000580
2: -13.5733652, 16.5934029, -13.6056004, 16.6104469, -29.5453949, 29.5545387
3: -13.8997135, 23.4499359, -13.9515963, 23.4827042, -36.9460526, 36.9514084
4: -18.6009102, 18.1296806, -18.6339378, 18.1508293, -36.7517395, 36.7636185
5: -16.0646915, 19.9914207, -16.1157341, 20.0209160, -36.0856094, 36.1071548
6: -25.9542885, 13.9975071, -25.9712334, 14.0043926, -39.9586792, 39.9687424
7: -23.3105659, 18.8703365, -23.3352032, 18.8847389, -42.1953049, 42.2055397
8: -20.6540108, 23.7185936, -20.6889095, 23.7419739, -44.3239746, 44.3310623
9: -14.7345972, 19.4566002, -14.7515001, 19.4735699, -34.2081680, 34.2080994
10: -29.7198868, 17.1218472, -29.7432785, 17.1647530, -46.8846397, 46.8651276
11: -33.7700424, 7.3874693, -33.7948112, 7.4344449, -41.2044868, 41.1822815
12: -27.9256020, 11.8657722, -27.9498138, 11.9085093, -39.3841400, 39.3780556
13: -18.0050201, 28.4597416, -18.0818996, 28.4898911, -46.4949112, 46.5416412
14: -56.5637512, -1.6029129, -56.5968933, -1.5535088, -49.9176178, 49.9331360
15: -21.7787457, 17.5714874, -21.7986679, 17.5829601, -39.3617058, 39.3701553
16: -33.0470963, 13.7149658, -33.0654526, 13.7452850, -46.7923813, 46.7804184
17: -62.9017258, 0.6558590, -62.9069405, 0.6746674, -62.0924988, 62.0779343
18: -34.8249512, 3.6362448, -34.8470612, 3.6921854, -36.8515015, 36.8464928
19: -27.2910538, 3.1184683, -27.3162518, 3.1502070, -30.4412613, 30.4347191
20: -19.1830025, 10.1529522, -19.1927299, 10.1798277, -28.7043533, 28.7021027
21: -31.7393398, 4.3484774, -31.7652302, 4.3809147, -36.1202545, 36.1137085
22: -32.1746292, 6.5283632, -32.1980209, 6.5604992, -38.3797607, 38.3797073
23: -23.3959522, 7.4463601, -23.4235439, 7.4934945, -30.8894463, 30.8699036
24: -28.0516357, 9.3704576, -28.0820580, 9.4201298, -37.4717636, 37.4525146
25: -21.9722672, 11.5682297, -21.9947929, 11.6059437, -33.5061798, 33.4986038
26: -34.8866653, 10.6693048, -34.9035873, 10.7253571, -43.7209930, 43.7134399
27: -28.7381935, 7.4818082, -28.7706680, 7.5342555, -36.2724495, 36.2524757
28: -22.4478683, 12.5668402, -22.4688148, 12.6069698, -35.0548401, 35.0356560
29: -34.3901787, 3.8866959, -34.4177399, 3.9196339, -38.3098145, 38.3044357
30: -25.8777885, 12.1668243, -25.8974838, 12.2018967, -38.0796852, 38.0643082
31: -34.2238045, 6.5450792, -34.2605400, 6.5891294, -40.8129349, 40.8056183
32: -20.6729469, 13.4103231, -20.6959629, 13.4316177, -34.1045647, 34.1062851
33: -30.0581818, 21.1895180, -30.1113205, 21.1824551, -51.0665588, 51.1402817
34: -28.8083038, 17.1048489, -28.8243389, 17.1325722, -45.9408760, 45.9291878
35: -25.8847160, 20.2991791, -25.9189186, 20.2949562, -46.1796722, 46.2180977
36: -24.5460129, 18.9626179, -24.5700111, 18.9739876, -43.4762802, 43.4894104
37: -44.6669960, 13.7353706, -44.7007408, 13.7645769, -58.1753540, 58.2007904
38: -33.0340424, 18.2813568, -33.0608749, 18.3124599, -51.3465042, 51.3422318
39: -34.6123810, 16.7891312, -34.6634903, 16.8050804, -51.2615585, 51.2974167
40: -34.5735245, 15.5460882, -34.5962677, 15.5608253, -49.6399078, 49.6489868
41: -24.5249023, 14.6357059, -24.5432453, 14.6595669, -39.1844711, 39.1789513
42: -16.4619370, 11.0684509, -16.4717216, 11.0849218, -27.5468597, 27.5401726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=118, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 753

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 573

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5069408, upper bound: 27.5796911
time: 54.45 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5412336, upper bound: 27.5839085
time: 45.44 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -37.0092010, 14.1817322, -37.0090218, 14.1845503, -51.0793304, 51.0635071
1: -19.7651443, 16.4568329, -19.7644958, 16.4580879, -36.2232323, 36.2213287
2: -13.6350822, 16.6165409, -13.6347227, 16.6163406, -29.6050034, 29.6102829
3: -14.0000639, 23.4911194, -13.9994936, 23.4909515, -37.0344696, 37.0435028
4: -18.6639099, 18.1597958, -18.6632462, 18.1595459, -36.8234558, 36.8230438
5: -16.1633549, 20.0278206, -16.1627960, 20.0275688, -36.1909256, 36.1906166
6: -25.9905510, 14.0096607, -25.9877510, 14.0102558, -40.0008087, 39.9974136
7: -23.3595295, 18.8891869, -23.3588028, 18.8891983, -42.2487259, 42.2479897
8: -20.7216225, 23.7521763, -20.7209644, 23.7525215, -44.3934326, 44.3973312
9: -14.7697306, 19.4881630, -14.7691765, 19.4885178, -34.2582474, 34.2573395
10: -29.7535095, 17.2103386, -29.7533417, 17.2065620, -46.9600716, 46.9636803
11: -33.8037643, 7.4758325, -33.8036423, 7.4758630, -41.2796288, 41.2794762
12: -27.9588947, 11.9470100, -27.9589024, 11.9466562, -39.4519043, 39.4443893
13: -18.1525230, 28.4970818, -18.1529217, 28.4970589, -46.6495819, 46.6500015
14: -56.6086044, -1.5087280, -56.6082916, -1.5089092, -50.0076218, 49.9797363
15: -21.8189011, 17.5930424, -21.8188515, 17.5926113, -39.4115143, 39.4118958
16: -33.0878868, 13.7841949, -33.0877533, 13.7801428, -46.8680305, 46.8719482
17: -62.9160919, 0.6907196, -62.9160614, 0.6906967, -62.1172409, 62.1515427
18: -34.8517685, 3.7449646, -34.8518677, 3.7446480, -36.9378586, 36.9174194
19: -27.3248425, 3.1811767, -27.3247566, 3.1806316, -30.5054741, 30.5059338
20: -19.1984768, 10.2040577, -19.1981697, 10.2038784, -28.7472839, 28.7399673
21: -31.7755642, 4.4119158, -31.7753677, 4.4115987, -36.1871643, 36.1872826
22: -32.2063484, 6.5910602, -32.2066917, 6.5907288, -38.4431763, 38.4375992
23: -23.4313431, 7.5378861, -23.4311543, 7.5371218, -30.9684639, 30.9690399
24: -28.0897808, 9.4671497, -28.0897751, 9.4666080, -37.5563889, 37.5569229
25: -22.0032177, 11.6424589, -22.0029984, 11.6418476, -33.5752716, 33.5701065
26: -34.9093742, 10.7759056, -34.9095154, 10.7754726, -43.7972183, 43.7820244
27: -28.7794762, 7.5830421, -28.7794838, 7.5830231, -36.3624992, 36.3625259
28: -22.4756126, 12.6447287, -22.4753838, 12.6440992, -35.1197128, 35.1201134
29: -34.4274979, 3.9502430, -34.4277954, 3.9500284, -38.3775253, 38.3780365
30: -25.9057426, 12.2338848, -25.9056339, 12.2336445, -38.1393890, 38.1395187
31: -34.2718048, 6.6321316, -34.2718201, 6.6315842, -40.9033890, 40.9036217
32: -20.7088089, 13.4527874, -20.7085381, 13.4527054, -34.1615143, 34.1613235
33: -30.1581364, 21.1912079, -30.1589203, 21.1907215, -51.1766510, 51.1878738
34: -28.8338318, 17.1593895, -28.8335915, 17.1588421, -45.9926758, 45.9929810
35: -25.9475479, 20.3013077, -25.9492264, 20.3006077, -46.2481537, 46.2505341
36: -24.5881462, 18.9849281, -24.5896606, 18.9848061, -43.5306778, 43.5327682
37: -44.7249832, 13.7928057, -44.7247391, 13.7917786, -58.2899170, 58.2852783
38: -33.0833168, 18.3406372, -33.0841293, 18.3404198, -51.4237366, 51.4247665
39: -34.7026901, 16.8220940, -34.7048721, 16.8215599, -51.3703918, 51.3733215
40: -34.6137199, 15.5753546, -34.6144257, 15.5752106, -49.6969910, 49.6984100
41: -24.5585098, 14.6829681, -24.5584393, 14.6830349, -39.2415466, 39.2414093
42: -16.4829731, 11.0986662, -16.4827538, 11.0992079, -27.5821800, 27.5814209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=118, inp2_unstable=118, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 753

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 573

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5591469, upper bound: 27.5875237
time: 61.33 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5908499, upper bound: 27.5908506
time: 54.69 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 118.01 seconds
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 118.01
Output dim: 13, lower bound: -27.5069408, upper bound: 27.5796911
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 118.01
Output dim: 13, lower bound: -27.5412336, upper bound: 27.5839085
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 118.01
Output dim: 13, lower bound: -27.5591469, upper bound: 27.5875237
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 118.01
Output dim: 13, lower bound: -27.5908499, upper bound: 27.5908506

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -36.9372520, 14.1443644, -36.9741859, 14.1663513, -50.9599304, 50.9759216
1: -19.7199478, 16.4209557, -19.7487373, 16.4416370, -36.1615829, 36.1696930
2: -13.5435352, 16.5810566, -13.5924797, 16.6064854, -29.5121689, 29.5302467
3: -13.8704529, 23.4376335, -13.9382172, 23.4769440, -36.9139099, 36.9340935
4: -18.5786667, 18.1282463, -18.6260452, 18.1463127, -36.7249794, 36.7542915
5: -16.0138302, 19.9680519, -16.0921288, 20.0153923, -36.0292206, 36.0601807
6: -25.9081955, 13.9728203, -25.9495850, 14.0010777, -39.9092712, 39.9224052
7: -23.2885113, 18.8614388, -23.3276329, 18.8800163, -42.1685257, 42.1890717
8: -20.6374798, 23.6956253, -20.6820183, 23.7311039, -44.2943878, 44.2998734
9: -14.7009592, 19.4159279, -14.7425213, 19.4543133, -34.1552734, 34.1584473
10: -29.6741791, 17.0389881, -29.7397728, 17.1251717, -46.7993507, 46.7787628
11: -33.7322617, 7.3311834, -33.7895546, 7.4075012, -41.1397629, 41.1207390
12: -27.8864689, 11.8001375, -27.9412575, 11.8773489, -39.3393936, 39.3109169
13: -17.9126949, 28.4387569, -18.0396481, 28.4842033, -46.3968964, 46.4784050
14: -56.4829636, -1.6948090, -56.5873566, -1.5990429, -49.7866974, 49.8276558
15: -21.7573814, 17.5407448, -21.7943134, 17.5684967, -39.3258781, 39.3350601
16: -33.0126266, 13.6678219, -33.0536346, 13.7228861, -46.7355118, 46.7214584
17: -62.8445358, 0.5954704, -62.9001617, 0.6452293, -62.0057373, 62.0097122
18: -34.7885132, 3.5767555, -34.8420410, 3.6632452, -36.7847214, 36.7795563
19: -27.2713089, 3.1112661, -27.3089924, 3.1471720, -30.4184799, 30.4202576
20: -19.1682854, 10.1305904, -19.1882362, 10.1696110, -28.6768494, 28.6742249
21: -31.7068462, 4.3231454, -31.7580795, 4.3687940, -36.0756416, 36.0812263
22: -32.1376877, 6.4878883, -32.1910248, 6.5409727, -38.3225021, 38.3321609
23: -23.3727512, 7.4096866, -23.4172840, 7.4763741, -30.8491249, 30.8269711
24: -28.0291748, 9.3479033, -28.0744801, 9.4098539, -37.4390297, 37.4223824
25: -21.9417381, 11.5301952, -21.9867401, 11.5893402, -33.4617233, 33.4530334
26: -34.8480301, 10.5951138, -34.8977890, 10.6895533, -43.6493301, 43.6357346
27: -28.7035007, 7.4327002, -28.7645607, 7.5102992, -36.2137985, 36.1972618
28: -22.4293861, 12.5348225, -22.4632530, 12.5929871, -35.0223732, 34.9980774
29: -34.3506775, 3.8433857, -34.4107590, 3.8991222, -38.2498016, 38.2541428
30: -25.8518410, 12.1301003, -25.8908634, 12.1852417, -38.0370827, 38.0209656
31: -34.1967735, 6.5351744, -34.2514229, 6.5848694, -40.7816429, 40.7865982
32: -20.6436787, 13.3978767, -20.6844387, 13.4270926, -34.0707703, 34.0823135
33: -29.9408779, 21.1445179, -30.0542469, 21.1781311, -50.9442749, 51.0384140
34: -28.7692719, 17.0836372, -28.8050938, 17.1268768, -45.8961487, 45.8887329
35: -25.7859669, 20.2570210, -25.8700542, 20.2925434, -46.0785103, 46.1270752
36: -24.4725742, 18.9404526, -24.5343399, 18.9713078, -43.4000092, 43.4312515
37: -44.5928688, 13.7105446, -44.6658669, 13.7604904, -58.0975189, 58.1413345
38: -32.9568329, 18.2558632, -33.0233498, 18.3056431, -51.2624741, 51.2792130
39: -34.4709930, 16.7552738, -34.5971680, 16.8002644, -51.1150131, 51.1966629
40: -34.5151062, 15.5214272, -34.5692863, 15.5572052, -49.5776825, 49.5971680
41: -24.4827957, 14.6142921, -24.5234661, 14.6555195, -39.1383133, 39.1377563
42: -16.4574928, 11.0471153, -16.4654388, 11.0780458, -27.5355377, 27.5125542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=117, inp2_unstable=118, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=333, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 753

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 683

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.4930211, upper bound: 27.5777001
time: 43.04 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5049501, upper bound: 27.5777001
time: 59.30 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -36.9698296, 14.1560373, -36.9872665, 14.1706409, -51.0053864, 51.0002213
1: -19.7422237, 16.4438400, -19.7535591, 16.4518566, -36.1940804, 36.1973991
2: -13.5709028, 16.5927143, -13.6041222, 16.6101265, -29.5380402, 29.5516891
3: -13.8928881, 23.4490356, -13.9479742, 23.4822788, -36.9449234, 36.9459839
4: -18.5932217, 18.1287975, -18.6303673, 18.1503944, -36.7436142, 36.7591629
5: -16.0612946, 19.9905930, -16.1141720, 20.0205154, -36.0818100, 36.1047668
6: -25.9501801, 13.9968777, -25.9692383, 14.0040798, -39.9542618, 39.9661179
7: -23.3081703, 18.8696384, -23.3340893, 18.8843880, -42.1925583, 42.2037277
8: -20.6498795, 23.7175179, -20.6868172, 23.7414684, -44.3270798, 44.3269882
9: -14.7334538, 19.4544067, -14.7509604, 19.4723396, -34.2057953, 34.2053680
10: -29.7189407, 17.1189175, -29.7428474, 17.1633720, -46.8823128, 46.8617630
11: -33.7691307, 7.3855805, -33.7943802, 7.4335637, -41.2026939, 41.1799622
12: -27.9246788, 11.8634033, -27.9493694, 11.9074049, -39.3816528, 39.3630676
13: -18.0020332, 28.4588699, -18.0804787, 28.4894962, -46.4915314, 46.5393486
14: -56.5629654, -1.6061592, -56.5965080, -1.5549889, -49.9150696, 49.8583565
15: -21.7781239, 17.5686512, -21.7984047, 17.5816307, -39.3597565, 39.3670578
16: -33.0456238, 13.7131824, -33.0647621, 13.7444334, -46.7900581, 46.7779465
17: -62.9010658, 0.6533699, -62.9066429, 0.6735268, -62.0905762, 62.0456314
18: -34.8242722, 3.6337862, -34.8467445, 3.6910124, -36.8479385, 36.8010559
19: -27.2899742, 3.1165042, -27.3157444, 3.1492734, -30.4392471, 30.4322491
20: -19.1823425, 10.1500168, -19.1924133, 10.1784420, -28.6976509, 28.6965446
21: -31.7382965, 4.3475318, -31.7647343, 4.3804703, -36.1187668, 36.1122665
22: -32.1737366, 6.5267630, -32.1976013, 6.5597291, -38.3781204, 38.3645897
23: -23.3949337, 7.4429617, -23.4230709, 7.4918590, -30.8867931, 30.8660316
24: -28.0504036, 9.3667622, -28.0814819, 9.4184093, -37.4688110, 37.4482422
25: -21.9711361, 11.5654125, -21.9942608, 11.6045990, -33.5029793, 33.4875221
26: -34.8856239, 10.6666412, -34.9030952, 10.7240791, -43.7187042, 43.6682549
27: -28.7373352, 7.4799738, -28.7702656, 7.5334034, -36.2707367, 36.2502403
28: -22.4470634, 12.5635347, -22.4684258, 12.6054106, -35.0524750, 35.0319595
29: -34.3893166, 3.8853436, -34.4173203, 3.9189968, -38.3083115, 38.3026657
30: -25.8766747, 12.1656036, -25.8969460, 12.2013102, -38.0779839, 38.0625496
31: -34.2225761, 6.5426464, -34.2599487, 6.5878735, -40.8104477, 40.8025970
32: -20.6710606, 13.4082279, -20.6950569, 13.4306335, -34.1016922, 34.1032867
33: -30.0538197, 21.1890182, -30.1092834, 21.1822052, -51.0403137, 51.1377029
34: -28.8048973, 17.1042042, -28.8227005, 17.1322422, -45.9371414, 45.9269028
35: -25.8809948, 20.2988625, -25.9171410, 20.2947941, -46.1757889, 46.2160034
36: -24.5433521, 18.9623260, -24.5687275, 18.9738159, -43.4716492, 43.4877777
37: -44.6636772, 13.7350464, -44.6991501, 13.7644234, -58.1552505, 58.1984634
38: -33.0311356, 18.2805958, -33.0595512, 18.3121071, -51.3432426, 51.3401489
39: -34.6072426, 16.7887115, -34.6610641, 16.8048592, -51.2543488, 51.2945328
40: -34.5709686, 15.5458031, -34.5949974, 15.5606813, -49.6334686, 49.6473160
41: -24.5223694, 14.6352501, -24.5420303, 14.6593761, -39.1817474, 39.1772804
42: -16.4608784, 11.0606995, -16.4712086, 11.0808105, -27.5416889, 27.5319080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=117, inp2_unstable=118, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 753

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 683

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.4930211, upper bound: 27.5536789
time: 54.07 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5392442, upper bound: 27.5819161
time: 60.21 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -36.9753036, 14.1646595, -36.9953918, 14.1777821, -51.0316772, 51.0292664
1: -19.7424088, 16.4311180, -19.7594719, 16.4466953, -36.1891022, 36.1905899
2: -13.6051798, 16.6039314, -13.6215601, 16.6123772, -29.5716934, 29.5857773
3: -13.9707241, 23.4785461, -13.9861021, 23.4852200, -37.0022507, 37.0257797
4: -18.6415787, 18.1576977, -18.6553364, 18.1550922, -36.7966690, 36.8130341
5: -16.1123695, 20.0044441, -16.1391487, 20.0220928, -36.1344604, 36.1435928
6: -25.9436855, 13.9848709, -25.9661102, 14.0069494, -39.9506340, 39.9509811
7: -23.3373566, 18.8798885, -23.3512154, 18.8845158, -42.2218704, 42.2311020
8: -20.7049599, 23.7285748, -20.7140827, 23.7416496, -44.3637466, 44.3656006
9: -14.7360048, 19.4468098, -14.7602301, 19.4692516, -34.2052574, 34.2070389
10: -29.7077904, 17.1266670, -29.7498474, 17.1669540, -46.8747444, 46.8765144
11: -33.7659225, 7.4194741, -33.7984009, 7.4489307, -41.2148514, 41.2178764
12: -27.9198341, 11.8812408, -27.9504051, 11.9154634, -39.4070816, 39.3769951
13: -18.0597248, 28.4759331, -18.1106319, 28.4913845, -46.5511093, 46.5865631
14: -56.5276718, -1.6007156, -56.5987473, -1.5544662, -49.8767166, 49.8740540
15: -21.7974052, 17.5615406, -21.8144569, 17.5781631, -39.3755684, 39.3759995
16: -33.0533905, 13.7368269, -33.0759583, 13.7577324, -46.8111229, 46.8127861
17: -62.8587036, 0.6302242, -62.9092789, 0.6611786, -62.0303116, 62.0832291
18: -34.8152542, 3.6852932, -34.8468781, 3.7157259, -36.8709641, 36.8502045
19: -27.3049011, 3.1739383, -27.3174725, 3.1775980, -30.4824982, 30.4914112
20: -19.1835785, 10.1816053, -19.1936932, 10.1936646, -28.7195396, 28.7119827
21: -31.7429371, 4.3865037, -31.7682114, 4.3994741, -36.1424103, 36.1547165
22: -32.1692886, 6.5504456, -32.1996994, 6.5711837, -38.3857880, 38.3898773
23: -23.4079933, 7.5010424, -23.4249153, 7.5199690, -30.9279633, 30.9259567
24: -28.0671272, 9.4444733, -28.0821819, 9.4563446, -37.5234718, 37.5266571
25: -21.9724503, 11.6042728, -21.9949303, 11.6252384, -33.5306854, 33.5244179
26: -34.8706284, 10.7015343, -34.9037247, 10.7396879, -43.7254715, 43.7039108
27: -28.7446938, 7.5338206, -28.7733784, 7.5590420, -36.3037338, 36.3071976
28: -22.4568520, 12.6125851, -22.4698181, 12.6301098, -35.0869598, 35.0824051
29: -34.3878860, 3.9068527, -34.4208641, 3.9294834, -38.3173676, 38.3277168
30: -25.8796387, 12.1969967, -25.8990154, 12.2169514, -38.0965881, 38.0960121
31: -34.2444801, 6.6221428, -34.2627029, 6.6273518, -40.8718338, 40.8821678
32: -20.6792450, 13.4402657, -20.6970310, 13.4481478, -34.1273918, 34.1372986
33: -30.0395432, 21.1461792, -30.1018143, 21.1864014, -51.0530243, 51.0859375
34: -28.7942772, 17.1381721, -28.8143845, 17.1531448, -45.9474220, 45.9525566
35: -25.8478107, 20.2591152, -25.9003105, 20.2981949, -46.1460037, 46.1594238
36: -24.5137863, 18.9627247, -24.5539932, 18.9821453, -43.4535294, 43.4745483
37: -44.6496468, 13.7679443, -44.6898727, 13.7877760, -58.2110748, 58.2256317
38: -33.0049629, 18.3150177, -33.0466042, 18.3335762, -51.3385391, 51.3616219
39: -34.5598602, 16.7881794, -34.6383362, 16.8167610, -51.2223206, 51.2723618
40: -34.5546112, 15.5506277, -34.5873489, 15.5715952, -49.6341095, 49.6465302
41: -24.5159950, 14.6614952, -24.5386581, 14.6789761, -39.1949692, 39.2001534
42: -16.4780827, 11.0772419, -16.4765167, 11.0923252, -27.5704079, 27.5537586

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=117, inp2_unstable=118, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 753

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 683

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5451459, upper bound: 27.5855350
time: 57.94 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5049501, upper bound: 27.5580133
time: 40.37 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -37.0079994, 14.1764488, -37.0084534, 14.1820965, -51.0771179, 51.0537338
1: -19.7647476, 16.4543419, -19.7642956, 16.4569283, -36.2216759, 36.2186356
2: -13.6326637, 16.6158447, -13.6332417, 16.6160088, -29.5976639, 29.6074295
3: -13.9932652, 23.4902115, -13.9958553, 23.4905396, -37.0333710, 37.0380516
4: -18.6562309, 18.1589394, -18.6596718, 18.1591492, -36.8153801, 36.8186111
5: -16.1599731, 20.0269585, -16.1612015, 20.0271835, -36.1871567, 36.1881599
6: -25.9864845, 14.0089912, -25.9857750, 14.0099564, -39.9964409, 39.9947662
7: -23.3571205, 18.8884926, -23.3576851, 18.8888588, -42.2459793, 42.2461777
8: -20.7174816, 23.7510643, -20.7188854, 23.7519798, -44.3965149, 44.3932648
9: -14.7685966, 19.4859810, -14.7686682, 19.4872932, -34.2558899, 34.2546501
10: -29.7525787, 17.2074089, -29.7529106, 17.2052040, -46.9577827, 46.9603195
11: -33.8028564, 7.4739609, -33.8032303, 7.4749813, -41.2778397, 41.2771912
12: -27.9579773, 11.9446507, -27.9584808, 11.9455404, -39.4494553, 39.4296341
13: -18.1495590, 28.4962406, -18.1515045, 28.4966583, -46.6462173, 46.6477432
14: -56.6078110, -1.5119553, -56.6079254, -1.5104160, -50.0051193, 49.9049110
15: -21.8182526, 17.5902081, -21.8185368, 17.5912685, -39.4095230, 39.4087448
16: -33.0863876, 13.7824173, -33.0870476, 13.7792988, -46.8656845, 46.8694649
17: -62.9153976, 0.6882420, -62.9157791, 0.6895142, -62.1153259, 62.1192551
18: -34.8511124, 3.7424870, -34.8515472, 3.7434978, -36.9342728, 36.8720245
19: -27.3237534, 3.1791892, -27.3242378, 3.1796989, -30.5034523, 30.5034275
20: -19.1978188, 10.2011242, -19.1978569, 10.2025080, -28.7405930, 28.7343903
21: -31.7745171, 4.4109697, -31.7749043, 4.4111400, -36.1856575, 36.1858749
22: -32.2054443, 6.5894632, -32.2062492, 6.5899830, -38.4415131, 38.4224701
23: -23.4303455, 7.5344934, -23.4306831, 7.5354800, -30.9658260, 30.9651756
24: -28.0885563, 9.4634495, -28.0891991, 9.4649067, -37.5534630, 37.5526505
25: -22.0020771, 11.6396313, -22.0024643, 11.6405067, -33.5720673, 33.5590134
26: -34.9083328, 10.7732220, -34.9090271, 10.7742100, -43.7949142, 43.7368355
27: -28.7786026, 7.5812221, -28.7790833, 7.5821829, -36.3607864, 36.3603058
28: -22.4748116, 12.6414270, -22.4750023, 12.6425514, -35.1173630, 35.1164284
29: -34.4265938, 3.9488735, -34.4273987, 3.9493866, -38.3759804, 38.3762741
30: -25.9046326, 12.2326679, -25.9051094, 12.2330494, -38.1376801, 38.1377792
31: -34.2705841, 6.6296835, -34.2712440, 6.6303530, -40.9009361, 40.9009285
32: -20.7069206, 13.4506750, -20.7076492, 13.4517059, -34.1586266, 34.1583252
33: -30.1537743, 21.1907234, -30.1568718, 21.1904945, -51.1504669, 51.1852264
34: -28.8304558, 17.1587391, -28.8319702, 17.1585274, -45.9889832, 45.9907074
35: -25.9437904, 20.3009605, -25.9474487, 20.3004532, -46.2442436, 46.2484093
36: -24.5854874, 18.9846077, -24.5884018, 18.9846649, -43.5260162, 43.5312042
37: -44.7216949, 13.7925053, -44.7231636, 13.7916393, -58.2698364, 58.2829056
38: -33.0804596, 18.3398418, -33.0827751, 18.3400364, -51.4204941, 51.4226151
39: -34.6975212, 16.8216820, -34.7024307, 16.8213863, -51.3631439, 51.3704376
40: -34.6111870, 15.5750885, -34.6131821, 15.5750847, -49.6905518, 49.6967468
41: -24.5559521, 14.6825571, -24.5572243, 14.6828365, -39.2387886, 39.2397804
42: -16.4819221, 11.0909367, -16.4822598, 11.0950966, -27.5770187, 27.5731964

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=117, inp2_unstable=118, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 753

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 683

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5768391, upper bound: 27.5888588
time: 53.67 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5049501, upper bound: 27.5777001
time: 178.13 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 233.99 seconds
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 233.99
Output dim: 13, lower bound: -27.4930211, upper bound: 27.5777001
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 233.99
Output dim: 13, lower bound: -27.5049501, upper bound: 27.5777001
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 233.99
Output dim: 13, lower bound: -27.4930211, upper bound: 27.5536789
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 233.99
Output dim: 13, lower bound: -27.5392442, upper bound: 27.5819161
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 233.99
Output dim: 13, lower bound: -27.5451459, upper bound: 27.5855350
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 233.99
Output dim: 13, lower bound: -27.5049501, upper bound: 27.5580133
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 233.99
Output dim: 13, lower bound: -27.5768391, upper bound: 27.5888588
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 233.99
Output dim: 13, lower bound: -27.5049501, upper bound: 27.5777001

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -36.9071808, 14.1375227, -36.9090233, 14.1308918, -50.8898087, 50.9035950
1: -19.6899757, 16.4172382, -19.6852646, 16.4094067, -36.0993805, 36.1025009
2: -13.5246382, 16.5776653, -13.5530481, 16.5878029, -29.4708481, 29.4872055
3: -13.8606930, 23.4348717, -13.9168692, 23.4706192, -36.8889847, 36.9074783
4: -18.5731335, 18.1252174, -18.6136951, 18.1368027, -36.7099380, 36.7389145
5: -15.9996033, 19.9632111, -16.0618496, 19.9986763, -35.9982796, 36.0250626
6: -25.8872604, 13.9676666, -25.9078693, 14.0051327, -39.8923950, 39.8755341
7: -23.2432976, 18.8573189, -23.2336845, 18.8412666, -42.0845642, 42.0910034
8: -20.6254749, 23.6908340, -20.6559105, 23.7054749, -44.2520981, 44.2679749
9: -14.6932898, 19.4120445, -14.7380552, 19.4463711, -34.1396599, 34.1501007
10: -29.6605492, 17.0327072, -29.7285595, 17.1105881, -46.7711372, 46.7612686
11: -33.6903458, 7.3248529, -33.7036591, 7.3561287, -41.0464745, 41.0285110
12: -27.8787384, 11.7879248, -27.9187012, 11.8510742, -39.3035278, 39.2646103
13: -17.9001579, 28.4177818, -17.9845848, 28.4417953, -46.3419533, 46.4023666
14: -56.4596252, -1.6973286, -56.5377045, -1.6102943, -49.7384415, 49.7743797
15: -21.7481518, 17.5149231, -21.7714710, 17.5168419, -39.2649918, 39.2863922
16: -32.9849281, 13.6634283, -32.9927673, 13.6907301, -46.6756592, 46.6561966
17: -62.8223648, 0.5901203, -62.8517036, 0.6180115, -61.9571762, 61.9557190
18: -34.7672119, 3.5661812, -34.7999573, 3.6292286, -36.7275543, 36.7285156
19: -27.2634678, 3.1073751, -27.2921181, 3.1349096, -30.3983765, 30.3994942
20: -19.1460381, 10.1249094, -19.1423798, 10.1514626, -28.6323700, 28.6199875
21: -31.6936054, 4.3183012, -31.7313385, 4.3494153, -36.0430222, 36.0496407
22: -32.1287613, 6.4762001, -32.1677322, 6.5161529, -38.2861633, 38.2922058
23: -23.3591652, 7.4034724, -23.3888626, 7.4527903, -30.8119545, 30.7923355
24: -28.0213661, 9.3421993, -28.0587120, 9.3928671, -37.4142342, 37.4009094
25: -21.9377937, 11.5209818, -21.9820099, 11.5707016, -33.4352951, 33.4368095
26: -34.8326416, 10.5812006, -34.8638573, 10.6702261, -43.6217880, 43.5932922
27: -28.6704330, 7.4220319, -28.6982059, 7.4637318, -36.1341629, 36.1202393
28: -22.4156227, 12.5313845, -22.4348946, 12.5861053, -35.0017281, 34.9662781
29: -34.3396721, 3.8399315, -34.3863945, 3.8839722, -38.2236443, 38.2263260
30: -25.8430405, 12.1262913, -25.8705444, 12.1729507, -38.0159912, 37.9968338
31: -34.1861801, 6.5271063, -34.2275467, 6.5635757, -40.7497559, 40.7546539
32: -20.6303520, 13.3938923, -20.6569233, 13.4352255, -34.0655785, 34.0508156
33: -29.9295120, 21.0934105, -29.9808369, 21.0750008, -50.8290100, 50.9132233
34: -28.7606773, 17.0506897, -28.7629776, 17.0578213, -45.8184967, 45.8136673
35: -25.7775497, 20.2116547, -25.8067360, 20.1988297, -45.9763794, 46.0183907
36: -24.4685440, 18.9337597, -24.5143890, 18.9565067, -43.3813095, 43.4046783
37: -44.5843353, 13.6777782, -44.6185760, 13.6926060, -58.0209503, 58.0612564
38: -32.9510422, 18.2448769, -33.0059891, 18.3036575, -51.2546997, 51.2508659
39: -34.4603577, 16.7135830, -34.5306168, 16.7149487, -51.0191269, 51.0884933
40: -34.5050659, 15.5133858, -34.5373955, 15.5539436, -49.5633545, 49.5561066
41: -24.4699135, 14.6081495, -24.4949684, 14.6614323, -39.1313477, 39.1031189
42: -16.4343605, 11.0405169, -16.4178104, 11.0558128, -27.4901733, 27.4583282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=117, inp2_unstable=117, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=333, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 642

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1719

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.4881298, upper bound: 27.5451578
time: 55.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.4928049, upper bound: 27.5765148
time: 44.97 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -36.9364815, 14.1439800, -36.9721260, 14.1653633, -50.9620667, 50.9726410
1: -19.7193546, 16.4206810, -19.7471924, 16.4408970, -36.1602516, 36.1678734
2: -13.5431499, 16.5806274, -13.5914183, 16.6054516, -29.5134048, 29.5280991
3: -13.8701649, 23.4372520, -13.9374084, 23.4759102, -36.9199753, 36.9307289
4: -18.5784187, 18.1277466, -18.6254253, 18.1450405, -36.7234573, 36.7531738
5: -16.0134697, 19.9677372, -16.0911865, 20.0146141, -36.0280838, 36.0589218
6: -25.9042358, 13.9725914, -25.9391537, 14.0005274, -39.9047623, 39.9117432
7: -23.2874737, 18.8611717, -23.3249359, 18.8792763, -42.1667480, 42.1861076
8: -20.6366901, 23.6953278, -20.6801529, 23.7302475, -44.2972260, 44.2970123
9: -14.7004108, 19.4147739, -14.7410793, 19.4512978, -34.1517105, 34.1558533
10: -29.6733093, 17.0358467, -29.7375660, 17.1167622, -46.7900696, 46.7734146
11: -33.7309456, 7.3309269, -33.7861633, 7.4068408, -41.1377869, 41.1170883
12: -27.8861046, 11.7998495, -27.9403381, 11.8765535, -39.3355408, 39.3219185
13: -17.9122887, 28.4362812, -18.0385532, 28.4785004, -46.3907890, 46.4748344
14: -56.4822273, -1.6952744, -56.5853729, -1.6002922, -49.7795563, 49.7873993
15: -21.7570076, 17.5389042, -21.7933464, 17.5636234, -39.3206329, 39.3322525
16: -33.0114441, 13.6673937, -33.0505829, 13.7217789, -46.7332230, 46.7179756
17: -62.8435745, 0.5948029, -62.8976212, 0.6435013, -62.0030212, 61.9940567
18: -34.7864761, 3.5759535, -34.8365402, 3.6611872, -36.7805977, 36.7568474
19: -27.2707462, 3.1110711, -27.3074532, 3.1466570, -30.4174042, 30.4185238
20: -19.1675797, 10.1303864, -19.1863861, 10.1691246, -28.6756134, 28.6625938
21: -31.7060280, 4.3229795, -31.7559414, 4.3683610, -36.0743904, 36.0789223
22: -32.1370163, 6.4863148, -32.1891899, 6.5371165, -38.3159637, 38.3315125
23: -23.3721771, 7.4093628, -23.4157219, 7.4756145, -30.8477917, 30.8250847
24: -28.0284004, 9.3476667, -28.0724754, 9.4092350, -37.4376373, 37.4201431
25: -21.9413109, 11.5278072, -21.9856377, 11.5830002, -33.4529419, 33.4484482
26: -34.8461723, 10.5944061, -34.8928604, 10.6876593, -43.6447525, 43.6417427
27: -28.7007256, 7.4322810, -28.7573833, 7.5092006, -36.2099266, 36.1896629
28: -22.4286842, 12.5345726, -22.4613838, 12.5923271, -35.0210114, 34.9959564
29: -34.3498192, 3.8431206, -34.4084587, 3.8984451, -38.2482643, 38.2515793
30: -25.8510475, 12.1299067, -25.8888168, 12.1846752, -38.0357208, 38.0187225
31: -34.1960564, 6.5348721, -34.2495232, 6.5841513, -40.7802086, 40.7843933
32: -20.6400967, 13.3976955, -20.6749020, 13.4265966, -34.0666924, 34.0725975
33: -29.9403191, 21.1433887, -30.0528755, 21.1751842, -50.9294739, 51.0359039
34: -28.7687111, 17.0826797, -28.8037109, 17.1243610, -45.8930740, 45.8863907
35: -25.7854767, 20.2560158, -25.8687344, 20.2898750, -46.0753517, 46.1247482
36: -24.4719620, 18.9401379, -24.5327988, 18.9704456, -43.3977585, 43.4293671
37: -44.5920029, 13.7099209, -44.6636047, 13.7589111, -58.0856552, 58.1384430
38: -32.9532242, 18.2552414, -33.0135307, 18.3040504, -51.2572746, 51.2687721
39: -34.4701424, 16.7544746, -34.5948868, 16.7981758, -51.1112518, 51.1935883
40: -34.5141029, 15.5212345, -34.5667572, 15.5566378, -49.5745163, 49.5943680
41: -24.4787712, 14.6140585, -24.5126877, 14.6549377, -39.1337090, 39.1267471
42: -16.4562550, 11.0467949, -16.4621849, 11.0772152, -27.5334702, 27.5089798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=117, inp2_unstable=117, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=333, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 750

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1719

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5001089, upper bound: 27.5169395
time: 50.54 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5047316, upper bound: 27.5765148
time: 52.12 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -36.9690552, 14.1556444, -36.9852180, 14.1696720, -51.0075150, 50.9969330
1: -19.7416515, 16.4435501, -19.7519875, 16.4510994, -36.1927490, 36.1955376
2: -13.5705109, 16.5923176, -13.6030693, 16.6090603, -29.5393181, 29.5495415
3: -13.8925781, 23.4486446, -13.9471378, 23.4812088, -36.9509659, 36.9426041
4: -18.5929928, 18.1283150, -18.6297379, 18.1491337, -36.7421265, 36.7580528
5: -16.0609512, 19.9902763, -16.1132202, 20.0197182, -36.0806694, 36.1034966
6: -25.9462166, 13.9966631, -25.9588051, 14.0035229, -39.9497375, 39.9554672
7: -23.3071556, 18.8693581, -23.3313828, 18.8836632, -42.1908188, 42.2007408
8: -20.6491013, 23.7171860, -20.6849747, 23.7406235, -44.3299255, 44.3241196
9: -14.7329121, 19.4532490, -14.7495070, 19.4693413, -34.2022552, 34.2027550
10: -29.7180824, 17.1157856, -29.7406502, 17.1549721, -46.8730545, 46.8564377
11: -33.7678146, 7.3853297, -33.7909813, 7.4329014, -41.2007141, 41.1763115
12: -27.9243011, 11.8631096, -27.9483929, 11.9066067, -39.3777924, 39.3740692
13: -18.0016136, 28.4564228, -18.0793800, 28.4837856, -46.4853973, 46.5358047
14: -56.5622253, -1.6066589, -56.5945282, -1.5562592, -49.9079208, 49.8180923
15: -21.7777290, 17.5667992, -21.7973976, 17.5767136, -39.3544426, 39.3641968
16: -33.0444565, 13.7127733, -33.0616989, 13.7433329, -46.7877884, 46.7744713
17: -62.9001122, 0.6527061, -62.9040413, 0.6717644, -62.0877914, 62.0299454
18: -34.8222351, 3.6329842, -34.8412247, 3.6889286, -36.8438110, 36.7783279
19: -27.2894039, 3.1162949, -27.3142071, 3.1487885, -30.4381924, 30.4305019
20: -19.1816425, 10.1498299, -19.1905479, 10.1779766, -28.6964417, 28.6848984
21: -31.7375031, 4.3473368, -31.7626286, 4.3800635, -36.1175652, 36.1099663
22: -32.1730652, 6.5252051, -32.1957741, 6.5558872, -38.3716202, 38.3639717
23: -23.3943634, 7.4426684, -23.4215126, 7.4911041, -30.8854675, 30.8641815
24: -28.0496502, 9.3665237, -28.0794945, 9.4177904, -37.4674416, 37.4460182
25: -21.9707336, 11.5630169, -21.9931602, 11.5982771, -33.4941788, 33.4829407
26: -34.8837738, 10.6658993, -34.8981552, 10.7221622, -43.7141190, 43.6742020
27: -28.7345448, 7.4795456, -28.7630863, 7.5323219, -36.2668686, 36.2426300
28: -22.4463501, 12.5633125, -22.4665680, 12.6047831, -35.0511322, 35.0298805
29: -34.3884277, 3.8850698, -34.4149933, 3.9183331, -38.3067627, 38.3000641
30: -25.8758717, 12.1653929, -25.8949127, 12.2007494, -38.0766220, 38.0603065
31: -34.2218781, 6.5423346, -34.2580643, 6.5871258, -40.8090057, 40.8003998
32: -20.6674995, 13.4080524, -20.6855373, 13.4301558, -34.0976562, 34.0935898
33: -30.0532913, 21.1879120, -30.1078949, 21.1792564, -51.0254974, 51.1351318
34: -28.8043365, 17.1032486, -28.8213043, 17.1297531, -45.9340897, 45.9245529
35: -25.8805046, 20.2978249, -25.9158535, 20.2921467, -46.1726532, 46.2136765
36: -24.5427818, 18.9619789, -24.5672092, 18.9729328, -43.4693375, 43.4859238
37: -44.6628189, 13.7344017, -44.6969376, 13.7627773, -58.1434326, 58.1955490
38: -33.0275307, 18.2799911, -33.0496864, 18.3105278, -51.3380585, 51.3296776
39: -34.6063919, 16.7879257, -34.6588135, 16.8028107, -51.2505798, 51.2914352
40: -34.5699806, 15.5455923, -34.5925064, 15.5600910, -49.6302795, 49.6445236
41: -24.5183487, 14.6350307, -24.5312805, 14.6587524, -39.1771011, 39.1663132
42: -16.4596252, 11.0603876, -16.4679565, 11.0799828, -27.5396080, 27.5283432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=117, inp2_unstable=117, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 753

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1719

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5343171, upper bound: 27.5493145
time: 40.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5380590, upper bound: 27.5807286
time: 98.02 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -36.9452286, 14.1578007, -36.9302368, 14.1423264, -50.9615555, 50.9569664
1: -19.7124462, 16.4274101, -19.6960087, 16.4145031, -36.1269493, 36.1234207
2: -13.5863037, 16.6005440, -13.5821581, 16.5936661, -29.5303688, 29.5427208
3: -13.9609451, 23.4757595, -13.9647980, 23.4788589, -36.9773407, 36.9991417
4: -18.6360683, 18.1546898, -18.6430111, 18.1455631, -36.7816315, 36.7976990
5: -16.0981445, 19.9995842, -16.1088696, 20.0053368, -36.1034813, 36.1084518
6: -25.9227772, 13.9797382, -25.9243889, 14.0110416, -39.9338188, 39.9041290
7: -23.2921562, 18.8757706, -23.2572994, 18.8457508, -42.1379089, 42.1330719
8: -20.6929893, 23.7238045, -20.6880016, 23.7160225, -44.3214569, 44.3336792
9: -14.7283497, 19.4429626, -14.7557621, 19.4613075, -34.1896591, 34.1987228
10: -29.6941452, 17.1204109, -29.7386265, 17.1523857, -46.8465309, 46.8590393
11: -33.7239914, 7.4131265, -33.7125282, 7.3975515, -41.1215439, 41.1256561
12: -27.9121056, 11.8690462, -27.9278145, 11.8891926, -39.3712311, 39.3306961
13: -18.0471725, 28.4549541, -18.0555210, 28.4489708, -46.4961433, 46.5104752
14: -56.5043221, -1.6032467, -56.5491219, -1.5657082, -49.8284454, 49.8207703
15: -21.7881737, 17.5357265, -21.7916164, 17.5265121, -39.3146858, 39.3273430
16: -33.0257111, 13.7323980, -33.0150833, 13.7255926, -46.7513046, 46.7474823
17: -62.8365593, 0.6248074, -62.8607788, 0.6339836, -61.9817429, 62.0292206
18: -34.7939491, 3.6746998, -34.8047447, 3.6817017, -36.8137932, 36.7991791
19: -27.2970505, 3.1700435, -27.3006134, 3.1653476, -30.4623985, 30.4706573
20: -19.1613350, 10.1759281, -19.1478252, 10.1754971, -28.6750755, 28.6577187
21: -31.7296925, 4.3816757, -31.7414780, 4.3800812, -36.1097717, 36.1231537
22: -32.1603394, 6.5387478, -32.1763916, 6.5463843, -38.3494644, 38.3499146
23: -23.3943958, 7.4948273, -23.3964787, 7.4963923, -30.8907890, 30.8913059
24: -28.0593204, 9.4387798, -28.0664139, 9.4393444, -37.4986649, 37.5051956
25: -21.9685116, 11.5951157, -21.9901962, 11.6066017, -33.5042725, 33.5081978
26: -34.8552246, 10.6875687, -34.8698006, 10.7203751, -43.6979218, 43.6614838
27: -28.7115936, 7.5231752, -28.7070255, 7.5124784, -36.2240715, 36.2302017
28: -22.4430847, 12.6091385, -22.4414749, 12.6232624, -35.0663452, 35.0506134
29: -34.3768845, 3.9033928, -34.3964844, 3.9143372, -38.2912216, 38.2998772
30: -25.8708363, 12.1932039, -25.8786964, 12.2046547, -38.0754929, 38.0718994
31: -34.2338715, 6.6140766, -34.2388115, 6.6060638, -40.8399353, 40.8498878
32: -20.6659050, 13.4362726, -20.6695251, 13.4562931, -34.1222000, 34.1057968
33: -30.0281448, 21.0950546, -30.0284042, 21.0832863, -50.9377289, 50.9607391
34: -28.7856846, 17.1052246, -28.7722168, 17.0841236, -45.8698082, 45.8774414
35: -25.8393841, 20.2137489, -25.8369427, 20.2044582, -46.0438423, 46.0506897
36: -24.5097694, 18.9560299, -24.5340233, 18.9673195, -43.4347992, 43.4479752
37: -44.6411095, 13.7352228, -44.6425362, 13.7198086, -58.1344147, 58.1455460
38: -32.9991684, 18.3040352, -33.0292435, 18.3315964, -51.3307648, 51.3332787
39: -34.5492516, 16.7464943, -34.5718842, 16.7314224, -51.1264496, 51.1642838
40: -34.5445938, 15.5425720, -34.5554810, 15.5683136, -49.6196976, 49.6054840
41: -24.5031128, 14.6553574, -24.5101910, 14.6849060, -39.1880188, 39.1655502
42: -16.4549503, 11.0706778, -16.4289017, 11.0700951, -27.5250454, 27.4995804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=117, inp2_unstable=117, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 753

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1719

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.5401128, upper bound: 27.5529033
time: 54.09 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5439495, upper bound: 27.5843349
time: 42.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -36.9779243, 14.1696033, -36.9433060, 14.1466208, -51.0070114, 50.9814034
1: -19.7347927, 16.4506321, -19.7008209, 16.4247131, -36.1595078, 36.1514511
2: -13.6137695, 16.6124668, -13.5938282, 16.5972900, -29.5563049, 29.5643730
3: -13.9835033, 23.4874172, -13.9745445, 23.4841633, -37.0084305, 37.0114326
4: -18.6507072, 18.1559372, -18.6473465, 18.1496258, -36.8003311, 36.8032837
5: -16.1457710, 20.0220985, -16.1309395, 20.0104256, -36.1561966, 36.1530380
6: -25.9655628, 14.0038357, -25.9440651, 14.0139942, -39.9795570, 39.9478989
7: -23.3119164, 18.8843746, -23.2637653, 18.8501091, -42.1620255, 42.1481400
8: -20.7055073, 23.7462883, -20.6928062, 23.7263298, -44.3542023, 44.3613739
9: -14.7609253, 19.4821014, -14.7641983, 19.4793644, -34.2402878, 34.2462997
10: -29.7389030, 17.2011280, -29.7417068, 17.1906166, -46.9295197, 46.9428329
11: -33.7609367, 7.4676142, -33.7173615, 7.4235792, -41.1845169, 41.1849747
12: -27.9502773, 11.9324408, -27.9359093, 11.9192963, -39.4135361, 39.3833580
13: -18.1369972, 28.4752522, -18.0963936, 28.4542294, -46.5912247, 46.5716476
14: -56.5844116, -1.5144958, -56.5582581, -1.5216827, -49.9568481, 49.8516350
15: -21.8090096, 17.5643749, -21.7956715, 17.5396194, -39.3486290, 39.3600464
16: -33.0586891, 13.7780218, -33.0261841, 13.7471743, -46.8058624, 46.8042068
17: -62.8932114, 0.6828423, -62.8672409, 0.6623192, -62.0667877, 62.0653000
18: -34.8297920, 3.7319145, -34.8094597, 3.7094584, -36.8770905, 36.8210030
19: -27.3159065, 3.1752896, -27.3073616, 3.1674690, -30.4833755, 30.4826508
20: -19.1755638, 10.1954393, -19.1519699, 10.1843452, -28.6961098, 28.6801186
21: -31.7612610, 4.4061346, -31.7481384, 4.3917766, -36.1530380, 36.1542740
22: -32.1964951, 6.5777688, -32.1828918, 6.5651627, -38.4051743, 38.3824844
23: -23.4167709, 7.5282869, -23.4022541, 7.5119133, -30.9286842, 30.9305420
24: -28.0807724, 9.4577446, -28.0734177, 9.4478855, -37.5286560, 37.5311623
25: -21.9981613, 11.6304798, -21.9977093, 11.6218739, -33.5456696, 33.5428009
26: -34.8929214, 10.7593117, -34.8750992, 10.7549200, -43.7673721, 43.6943741
27: -28.7455387, 7.5705905, -28.7127380, 7.5355949, -36.2811356, 36.2833290
28: -22.4610214, 12.6379700, -22.4466629, 12.6356907, -35.0967102, 35.0846329
29: -34.4155998, 3.9454136, -34.4030457, 3.9342117, -38.3498116, 38.3484573
30: -25.8958378, 12.2288685, -25.8847828, 12.2207451, -38.1165848, 38.1136513
31: -34.2599716, 6.6216478, -34.2473526, 6.6090555, -40.8690262, 40.8689995
32: -20.6936131, 13.4466801, -20.6801491, 13.4598312, -34.1534424, 34.1268311
33: -30.1424389, 21.1396236, -30.0834179, 21.0873547, -51.0351868, 51.0600204
34: -28.8218918, 17.1257763, -28.7898331, 17.0894947, -45.9113846, 45.9156113
35: -25.9353600, 20.2555885, -25.8840599, 20.2067223, -46.1420822, 46.1396484
36: -24.5814495, 18.9779186, -24.5684319, 18.9698372, -43.5073166, 43.5046005
37: -44.7131653, 13.7597342, -44.6758804, 13.7237129, -58.1932220, 58.2028275
38: -33.0746384, 18.3288517, -33.0654144, 18.3380451, -51.4126816, 51.3942642
39: -34.6869659, 16.7800026, -34.6359367, 16.7360668, -51.2672882, 51.2623444
40: -34.6011887, 15.5670099, -34.5812416, 15.5717983, -49.6762085, 49.6556778
41: -24.5430908, 14.6764193, -24.5287590, 14.6887627, -39.2318535, 39.2051773
42: -16.4587898, 11.0843458, -16.4346409, 11.0728674, -27.5316582, 27.5189857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=117, inp2_unstable=117, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 753

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1719

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5718012, upper bound: 27.5562326
time: 50.96 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.4881298, upper bound: 27.5253887
time: 37.67 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -37.0072556, 14.1760502, -37.0063705, 14.1811142, -51.0792694, 51.0504532
1: -19.7641735, 16.4540577, -19.7627449, 16.4561729, -36.2203445, 36.2168045
2: -13.6322670, 16.6154518, -13.6321907, 16.6149502, -29.5989456, 29.6052971
3: -13.9929581, 23.4898243, -13.9950342, 23.4894485, -37.0394363, 37.0346298
4: -18.6560059, 18.1584778, -18.6590233, 18.1578636, -36.8138695, 36.8175011
5: -16.1596146, 20.0266457, -16.1602745, 20.0263824, -36.1859970, 36.1869202
6: -25.9825001, 14.0087738, -25.9753284, 14.0093880, -39.9918900, 39.9841003
7: -23.3561039, 18.8882046, -23.3549786, 18.8881493, -42.2442551, 42.2431831
8: -20.7167015, 23.7507591, -20.7170105, 23.7511063, -44.3993912, 44.3903961
9: -14.7680454, 19.4848251, -14.7672062, 19.4842949, -34.2523422, 34.2520294
10: -29.7517509, 17.2042694, -29.7507153, 17.1967945, -46.9485474, 46.9549866
11: -33.8015366, 7.4737072, -33.7998428, 7.4743390, -41.2758751, 41.2735519
12: -27.9576015, 11.9443731, -27.9575043, 11.9447651, -39.4456024, 39.4406471
13: -18.1491318, 28.4937687, -18.1504021, 28.4909515, -46.6400833, 46.6441727
14: -56.6070633, -1.5124626, -56.6059227, -1.5117111, -49.9979706, 49.8646507
15: -21.8178711, 17.5883560, -21.8175507, 17.5863724, -39.4042435, 39.4059067
16: -33.0852356, 13.7819853, -33.0839882, 13.7781763, -46.8634109, 46.8659744
17: -62.9144211, 0.6876106, -62.9131660, 0.6877842, -62.1125946, 62.1035309
18: -34.8490601, 3.7416887, -34.8460312, 3.7414398, -36.9301300, 36.8492928
19: -27.3231888, 3.1789880, -27.3227081, 3.1792073, -30.5023956, 30.5016956
20: -19.1971092, 10.2009344, -19.1959991, 10.2020330, -28.7393875, 28.7227287
21: -31.7737389, 4.4108005, -31.7727451, 4.4107299, -36.1844673, 36.1835442
22: -32.2047615, 6.5879240, -32.2044449, 6.5861168, -38.4350128, 38.4218445
23: -23.4297638, 7.5341830, -23.4291134, 7.5347328, -30.9644966, 30.9632969
24: -28.0878105, 9.4632092, -28.0872040, 9.4642744, -37.5520859, 37.5504150
25: -22.0016747, 11.6372738, -22.0013695, 11.6341677, -33.5633011, 33.5544586
26: -34.9064484, 10.7725115, -34.9041138, 10.7723122, -43.7903290, 43.7428093
27: -28.7758255, 7.5807781, -28.7718964, 7.5810990, -36.3569260, 36.3526764
28: -22.4741001, 12.6411819, -22.4731407, 12.6419468, -35.1160469, 35.1143227
29: -34.4257355, 3.9486074, -34.4250832, 3.9487190, -38.3744545, 38.3736916
30: -25.9038200, 12.2324810, -25.9030476, 12.2324810, -38.1362991, 38.1355286
31: -34.2698746, 6.6293955, -34.2693748, 6.6296253, -40.8994980, 40.8964005
32: -20.7033596, 13.4504967, -20.6981239, 13.4512177, -34.1545792, 34.1486206
33: -30.1532516, 21.1896038, -30.1554832, 21.1875362, -51.1356354, 51.1827011
34: -28.8299103, 17.1578121, -28.8305321, 17.1560211, -45.9859314, 45.9883423
35: -25.9432907, 20.2999535, -25.9460945, 20.2978001, -46.2410889, 46.2460480
36: -24.5849209, 18.9842911, -24.5868549, 18.9837685, -43.5237808, 43.5293045
37: -44.7208328, 13.7919025, -44.7209473, 13.7900190, -58.2580109, 58.2800369
38: -33.0768356, 18.3392563, -33.0729332, 18.3384724, -51.4153061, 51.4121895
39: -34.6966858, 16.8208923, -34.7001724, 16.8193016, -51.3594131, 51.3673782
40: -34.6102371, 15.5748739, -34.6106262, 15.5744886, -49.6873550, 49.6939926
41: -24.5519600, 14.6823177, -24.5464592, 14.6822186, -39.2341766, 39.2287750
42: -16.4806862, 11.0906229, -16.4790039, 11.0942717, -27.5749588, 27.5696259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=117, inp2_unstable=117, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 753

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1719

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5838454, upper bound: 27.5562326
time: 50.59 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -27.5876675, upper bound: 27.5876679
time: 55.12 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 107.93 seconds
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 107.93
Output dim: 13, lower bound: -27.4881298, upper bound: 27.5451578
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 107.93
Output dim: 13, lower bound: -27.4928049, upper bound: 27.5765148
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 107.93
Output dim: 13, lower bound: -27.5001089, upper bound: 27.5169395
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 107.93
Output dim: 13, lower bound: -27.5047316, upper bound: 27.5765148
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 107.93
Output dim: 13, lower bound: -27.5343171, upper bound: 27.5493145
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 107.93
Output dim: 13, lower bound: -27.5380590, upper bound: 27.5807286
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 107.93
Output dim: 13, lower bound: -27.5401128, upper bound: 27.5529033
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 107.93
Output dim: 13, lower bound: -27.5439495, upper bound: 27.5843349
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 107.93
Output dim: 13, lower bound: -27.5718012, upper bound: 27.5562326
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 107.93
Output dim: 13, lower bound: -27.4881298, upper bound: 27.5253887
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 107.93
Output dim: 13, lower bound: -27.5838454, upper bound: 27.5562326
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 107.93
Output dim: 13, lower bound: -27.5876675, upper bound: 27.5876679

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -36.9052277, 14.1350231, -36.9084549, 14.1301575, -50.8870850, 50.8998833
1: -19.6891880, 16.4145470, -19.6850376, 16.4086151, -36.0978012, 36.0995865
2: -13.5240746, 16.5753975, -13.5528784, 16.5871086, -29.4695625, 29.4825096
3: -13.8595848, 23.4313259, -13.9165382, 23.4695549, -36.8911819, 36.9029770
4: -18.5723877, 18.1228924, -18.6134548, 18.1360817, -36.7084694, 36.7363472
5: -15.9990158, 19.9607487, -16.0616760, 19.9979591, -35.9969749, 36.0224228
6: -25.8827057, 13.9663830, -25.9065132, 14.0047703, -39.8874741, 39.8728943
7: -23.2424831, 18.8543072, -23.2334633, 18.8403778, -42.0828629, 42.0877686
8: -20.6246929, 23.6872749, -20.6556892, 23.7044163, -44.2510300, 44.2639999
9: -14.6902399, 19.4106998, -14.7371521, 19.4459648, -34.1362038, 34.1478500
10: -29.6562004, 17.0306091, -29.7272644, 17.1099472, -46.7661476, 46.7578735
11: -33.6871567, 7.3215904, -33.7027016, 7.3551717, -41.0423279, 41.0242920
12: -27.8759842, 11.7861471, -27.9179096, 11.8505459, -39.2996902, 39.2707253
13: -17.8977890, 28.4154472, -17.9838676, 28.4411011, -46.3388901, 46.3993149
14: -56.4565659, -1.7016449, -56.5367889, -1.6115570, -49.7340851, 49.7371292
15: -21.7466412, 17.5085220, -21.7710419, 17.5149460, -39.2615891, 39.2795639
16: -32.9779701, 13.6618662, -32.9906998, 13.6902828, -46.6682510, 46.6525650
17: -62.8188477, 0.5845718, -62.8506241, 0.6163216, -61.9520111, 61.9326096
18: -34.7648544, 3.5647163, -34.7992363, 3.6287785, -36.7234001, 36.7343712
19: -27.2560234, 3.1064229, -27.2898445, 3.1346517, -30.3906746, 30.3962669
20: -19.1448040, 10.1230783, -19.1420040, 10.1509056, -28.6305771, 28.6161575
21: -31.6895294, 4.3169327, -31.7301025, 4.3490129, -36.0385437, 36.0470352
22: -32.1271973, 6.4728622, -32.1672668, 6.5151606, -38.2835770, 38.2874451
23: -23.3569355, 7.4025612, -23.3881874, 7.4525275, -30.8094635, 30.7907486
24: -28.0194778, 9.3411570, -28.0581284, 9.3925495, -37.4120255, 37.3992844
25: -21.9362144, 11.5178604, -21.9815254, 11.5697746, -33.4328003, 33.4335213
26: -34.8305664, 10.5767574, -34.8632202, 10.6689224, -43.6183319, 43.5900383
27: -28.6673508, 7.4208798, -28.6972866, 7.4633760, -36.1307259, 36.1181679
28: -22.4143372, 12.5304594, -22.4344940, 12.5858393, -35.0001755, 34.9649544
29: -34.3382454, 3.8380823, -34.3859634, 3.8834000, -38.2216454, 38.2240448
30: -25.8415947, 12.1204472, -25.8700867, 12.1712208, -38.0128174, 37.9905319
31: -34.1765633, 6.5255938, -34.2246437, 6.5631380, -40.7397003, 40.7502365
32: -20.6267891, 13.3919868, -20.6558685, 13.4346638, -34.0614548, 34.0478554
33: -29.9256172, 21.0925102, -29.9796619, 21.0747452, -50.8168716, 50.9111176
34: -28.7574940, 17.0498562, -28.7620010, 17.0576057, -45.8151016, 45.8118591
35: -25.7746010, 20.2112904, -25.8058395, 20.1987286, -45.9733276, 46.0171280
36: -24.4661255, 18.9332848, -24.5135765, 18.9563580, -43.3760605, 43.4033813
37: -44.5785065, 13.6772614, -44.6168594, 13.6924067, -58.0046387, 58.0589218
38: -32.9480743, 18.2436523, -33.0051193, 18.3032761, -51.2513504, 51.2487717
39: -34.4559631, 16.7123318, -34.5292931, 16.7145576, -51.0113831, 51.0859489
40: -34.5003967, 15.5123777, -34.5359917, 15.5536232, -49.5532455, 49.5536957
41: -24.4663734, 14.6066704, -24.4939289, 14.6609821, -39.1273575, 39.1006012
42: -16.4298801, 11.0390501, -16.4165039, 11.0553732, -27.4852524, 27.4555550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=117, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=333, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=42, inp2_unstable=42, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 753

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 606

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.4776269, upper bound: 27.5038578
time: 54.93 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 13, lower bound: -27.4906923, upper bound: 27.5474383
time: 170.55 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 227.72 seconds
IS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 227.72
Output dim: 13, lower bound: -27.4776269, upper bound: 27.5038578
IS_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 227.72
Output dim: 13, lower bound: -27.4906923, upper bound: 27.5474383
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 227.72
Output dim: 13, lower bound: -27.5047316, upper bound: 27.5765148
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 227.72
Output dim: 13, lower bound: -27.5380590, upper bound: 27.5807286
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 227.72
Output dim: 13, lower bound: -27.5439495, upper bound: 27.5843349
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 227.72
Output dim: 13, lower bound: -27.5718012, upper bound: 27.5562326
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 227.72
Output dim: 13, lower bound: -27.5838454, upper bound: 27.5562326
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 227.72
Output dim: 13, lower bound: -27.5876675, upper bound: 27.5876679

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 106.92 + 1915.68 = 2022.60 seconds
