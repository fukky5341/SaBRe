## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 0)
Time budget: 1800 seconds
Split limit: 100
Threshold: 27.5202515007


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=362, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-47.0727310, 16.0515289, -47.0727310, 16.0515289, -63.1242599, 63.1242599)
1: (-25.2549191, 18.6986504, -25.2549191, 18.6986504, -43.7340012, 43.7340050)
2: (-21.5161400, 17.9457283, -21.5161400, 17.9457283, -39.4324722, 39.4324760)
3: (-25.0356750, 17.5100574, -25.0356750, 17.5100574, -42.5457306, 42.5457306)
4: (-28.3546085, 20.1954041, -28.3546085, 20.1954041, -48.5500107, 48.5500107)
5: (-23.6489449, 19.8753014, -23.6489449, 19.8753014, -43.5242462, 43.5242462)
6: (-33.7565460, 18.9100819, -33.7565460, 18.9100819, -51.8840103, 51.8840065)
7: (-30.3117294, 21.4584789, -30.3117294, 21.4584789, -51.2755661, 51.2755661)
8: (-32.2651901, 24.0401077, -32.2651901, 24.0401077, -56.3052979, 56.3052979)
9: (-26.7325325, 17.7273445, -26.7325325, 17.7273445, -44.4598770, 44.4598770)
10: (-36.5454521, 26.0213966, -36.5454521, 26.0213966, -62.5668488, 62.5668488)
11: (-27.4929466, 25.1164017, -27.4929466, 25.1164017, -52.6093483, 52.6093483)
12: (-35.1603775, 22.0571594, -35.1603775, 22.0571594, -55.8131561, 55.8131561)
13: (-40.0848770, 22.5472832, -40.0848770, 22.5472832, -62.6321602, 62.6321602)
14: (-56.5082397, 13.5440235, -56.5082397, 13.5440235, -70.0522614, 70.0522614)
15: (-28.3149033, 19.6139297, -28.3149033, 19.6139297, -47.9288330, 47.9288330)
16: (-34.9394836, 22.3987980, -34.9394836, 22.3987980, -57.3382797, 57.3382797)
17: (-51.8795128, 27.0620499, -51.8795128, 27.0620499, -78.9415588, 78.9415588)
18: (-30.0645676, 26.9229984, -30.0645676, 26.9229984, -56.9875641, 56.9875641)
19: (-17.7139587, 17.6513634, -17.7139587, 17.6513634, -35.2683220, 35.2683182)
20: (-20.0517960, 17.4920921, -20.0517960, 17.4920921, -37.5438881, 37.5438881)
21: (-24.4986229, 22.8401794, -24.4986229, 22.8401794, -47.3388023, 47.3388023)
22: (-25.5391426, 21.6915550, -25.5391426, 21.6915550, -47.2306976, 47.2306976)
23: (-18.7391624, 21.6828156, -18.7391624, 21.6828156, -40.4219780, 40.4219780)
24: (-22.4943600, 22.3042927, -22.4943600, 22.3042927, -44.7986526, 44.7986526)
25: (-19.4939919, 24.6075783, -19.4939919, 24.6075783, -44.1015701, 44.1015701)
26: (-33.8213387, 29.5184746, -33.8213387, 29.5184746, -63.3398132, 63.3398132)
27: (-24.1218414, 22.8650970, -24.1218414, 22.8650970, -46.9869385, 46.9869385)
28: (-18.7543030, 25.4579201, -18.7543030, 25.4579201, -44.0786629, 44.0786667)
29: (-25.6286564, 24.3507462, -25.6286564, 24.3507462, -49.9794006, 49.9794006)
30: (-25.7671089, 25.4406376, -25.7671089, 25.4406376, -51.2077484, 51.2077484)
31: (-21.9189606, 23.8200073, -21.9189606, 23.8200073, -45.7389679, 45.7389679)
32: (-36.1053810, 14.9813128, -36.1053810, 14.9813128, -49.6482010, 49.6482048)
33: (-50.8679428, 21.5127850, -50.8679428, 21.5127850, -69.9246216, 69.9246216)
34: (-50.5968628, 10.0608883, -50.5968628, 10.0608883, -56.8147430, 56.8147469)
35: (-42.9593925, 16.9696846, -42.9593925, 16.9696846, -57.5810356, 57.5810318)
36: (-41.9535332, 18.6489067, -41.9535332, 18.6489067, -60.5230408, 60.5230408)
37: (-55.7414169, 13.1964712, -55.7414169, 13.1964712, -67.8719940, 67.8720016)
38: (-52.8991852, 15.9395485, -52.8991852, 15.9395485, -68.8387299, 68.8387299)
39: (-61.7800980, 18.2122650, -61.7800980, 18.2122650, -79.5897675, 79.5897675)
40: (-48.4437256, 11.8220787, -48.4437256, 11.8220787, -59.2941208, 59.2941132)
41: (-35.9136467, 18.6692276, -35.9136467, 18.6692276, -53.4173737, 53.4173737)
42: (-26.5274906, 13.2349625, -26.5274906, 13.2349625, -38.2487335, 38.2487411)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.69 + 81.60 = 84.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 31, lower bound: -27.5477993, upper bound: 27.5477993

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1555

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1621

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5458286, upper bound: 27.5421916
time: 59.49 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5421916, upper bound: 27.5458285
time: 53.20 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 112.71 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 112.71
Output dim: 31, lower bound: -27.5458286, upper bound: 27.5421916
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 112.71
Output dim: 31, lower bound: -27.5421916, upper bound: 27.5458285

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -47.0727310, 16.0515289, -47.0727310, 16.0515289, -63.1242599, 63.1242599
1: -25.2549191, 18.6986504, -25.2549191, 18.6986504, -43.7315140, 43.7313347
2: -21.5161400, 17.9457283, -21.5161400, 17.9457283, -39.4268188, 39.4260674
3: -25.0356750, 17.5100574, -25.0356750, 17.5100574, -42.5457306, 42.5457306
4: -28.3546085, 20.1954041, -28.3546085, 20.1954041, -48.5500107, 48.5500107
5: -23.6489449, 19.8753014, -23.6489449, 19.8753014, -43.5242462, 43.5242462
6: -33.7565460, 18.9100819, -33.7565460, 18.9100819, -51.8716431, 51.8724022
7: -30.3117294, 21.4584789, -30.3117294, 21.4584789, -51.2691574, 51.2684593
8: -32.2651901, 24.0401077, -32.2651901, 24.0401077, -56.3052979, 56.3052979
9: -26.7325325, 17.7273445, -26.7325325, 17.7273445, -44.4598770, 44.4598770
10: -36.5454521, 26.0213966, -36.5454521, 26.0213966, -62.5668488, 62.5668488
11: -27.4929466, 25.1164017, -27.4929466, 25.1164017, -52.6093483, 52.6093483
12: -35.1603775, 22.0571594, -35.1603775, 22.0571594, -55.7992020, 55.8007736
13: -40.0848770, 22.5472832, -40.0848770, 22.5472832, -62.6321602, 62.6321602
14: -56.5082397, 13.5440235, -56.5082397, 13.5440235, -70.0522614, 70.0522614
15: -28.3149033, 19.6139297, -28.3149033, 19.6139297, -47.9288330, 47.9288330
16: -34.9394836, 22.3987980, -34.9394836, 22.3987980, -57.3382797, 57.3382797
17: -51.8795128, 27.0620499, -51.8795128, 27.0620499, -78.9415588, 78.9415588
18: -30.0645676, 26.9229984, -30.0645676, 26.9229984, -56.9875641, 56.9875641
19: -17.7139587, 17.6513634, -17.7139587, 17.6513634, -35.2648239, 35.2642326
20: -20.0517960, 17.4920921, -20.0517960, 17.4920921, -37.5438881, 37.5438881
21: -24.4986229, 22.8401794, -24.4986229, 22.8401794, -47.3388023, 47.3388023
22: -25.5391426, 21.6915550, -25.5391426, 21.6915550, -47.2306976, 47.2306976
23: -18.7391624, 21.6828156, -18.7391624, 21.6828156, -40.4219780, 40.4219780
24: -22.4943600, 22.3042927, -22.4943600, 22.3042927, -44.7986526, 44.7986526
25: -19.4939919, 24.6075783, -19.4939919, 24.6075783, -44.1015701, 44.1015701
26: -33.8213387, 29.5184746, -33.8213387, 29.5184746, -63.3398132, 63.3398132
27: -24.1218414, 22.8650970, -24.1218414, 22.8650970, -46.9869385, 46.9869385
28: -18.7543030, 25.4579201, -18.7543030, 25.4579201, -44.0821991, 44.0818787
29: -25.6286564, 24.3507462, -25.6286564, 24.3507462, -49.9794006, 49.9794006
30: -25.7671089, 25.4406376, -25.7671089, 25.4406376, -51.2077484, 51.2077484
31: -21.9189606, 23.8200073, -21.9189606, 23.8200073, -45.7389679, 45.7389679
32: -36.1053810, 14.9813128, -36.1053810, 14.9813128, -49.6373444, 49.6382179
33: -50.8679428, 21.5127850, -50.8679428, 21.5127850, -69.9095154, 69.9116669
34: -50.5968628, 10.0608883, -50.5968628, 10.0608883, -56.7858047, 56.7890167
35: -42.9593925, 16.9696846, -42.9593925, 16.9696846, -57.5651474, 57.5668411
36: -41.9535332, 18.6489067, -41.9535332, 18.6489067, -60.5202179, 60.5203972
37: -55.7414169, 13.1964712, -55.7414169, 13.1964712, -67.8623352, 67.8636093
38: -52.8991852, 15.9395485, -52.8991852, 15.9395485, -68.8387299, 68.8387299
39: -61.7800980, 18.2122650, -61.7800980, 18.2122650, -79.5838776, 79.5844040
40: -48.4437256, 11.8220787, -48.4437256, 11.8220787, -59.2850189, 59.2859268
41: -35.9136467, 18.6692276, -35.9136467, 18.6692276, -53.4083557, 53.4089584
42: -26.5274906, 13.2349625, -26.5274906, 13.2349625, -38.2326050, 38.2336197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=362, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 571

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1705

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5454011, upper bound: 27.5391322
time: 60.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5427677, upper bound: 27.5417687
time: 59.11 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -47.0727310, 16.0515289, -47.0727310, 16.0515289, -63.1242599, 63.1242599
1: -25.2549191, 18.6986504, -25.2549191, 18.6986504, -43.7313309, 43.7315178
2: -21.5161400, 17.9457283, -21.5161400, 17.9457283, -39.4260712, 39.4268150
3: -25.0356750, 17.5100574, -25.0356750, 17.5100574, -42.5457306, 42.5457306
4: -28.3546085, 20.1954041, -28.3546085, 20.1954041, -48.5500107, 48.5500107
5: -23.6489449, 19.8753014, -23.6489449, 19.8753014, -43.5242462, 43.5242462
6: -33.7565460, 18.9100819, -33.7565460, 18.9100819, -51.8724060, 51.8716469
7: -30.3117294, 21.4584789, -30.3117294, 21.4584789, -51.2684555, 51.2691536
8: -32.2651901, 24.0401077, -32.2651901, 24.0401077, -56.3052979, 56.3052979
9: -26.7325325, 17.7273445, -26.7325325, 17.7273445, -44.4598770, 44.4598770
10: -36.5454521, 26.0213966, -36.5454521, 26.0213966, -62.5668488, 62.5668488
11: -27.4929466, 25.1164017, -27.4929466, 25.1164017, -52.6093483, 52.6093483
12: -35.1603775, 22.0571594, -35.1603775, 22.0571594, -55.8007736, 55.7992020
13: -40.0848770, 22.5472832, -40.0848770, 22.5472832, -62.6321602, 62.6321602
14: -56.5082397, 13.5440235, -56.5082397, 13.5440235, -70.0522614, 70.0522614
15: -28.3149033, 19.6139297, -28.3149033, 19.6139297, -47.9288330, 47.9288330
16: -34.9394836, 22.3987980, -34.9394836, 22.3987980, -57.3382797, 57.3382797
17: -51.8795128, 27.0620499, -51.8795128, 27.0620499, -78.9415588, 78.9415588
18: -30.0645676, 26.9229984, -30.0645676, 26.9229984, -56.9875641, 56.9875641
19: -17.7139587, 17.6513634, -17.7139587, 17.6513634, -35.2642288, 35.2648239
20: -20.0517960, 17.4920921, -20.0517960, 17.4920921, -37.5438881, 37.5438881
21: -24.4986229, 22.8401794, -24.4986229, 22.8401794, -47.3388023, 47.3388023
22: -25.5391426, 21.6915550, -25.5391426, 21.6915550, -47.2306976, 47.2306976
23: -18.7391624, 21.6828156, -18.7391624, 21.6828156, -40.4219780, 40.4219780
24: -22.4943600, 22.3042927, -22.4943600, 22.3042927, -44.7986526, 44.7986526
25: -19.4939919, 24.6075783, -19.4939919, 24.6075783, -44.1015701, 44.1015701
26: -33.8213387, 29.5184746, -33.8213387, 29.5184746, -63.3398132, 63.3398132
27: -24.1218414, 22.8650970, -24.1218414, 22.8650970, -46.9869385, 46.9869385
28: -18.7543030, 25.4579201, -18.7543030, 25.4579201, -44.0818787, 44.0821953
29: -25.6286564, 24.3507462, -25.6286564, 24.3507462, -49.9794006, 49.9794006
30: -25.7671089, 25.4406376, -25.7671089, 25.4406376, -51.2077484, 51.2077484
31: -21.9189606, 23.8200073, -21.9189606, 23.8200073, -45.7389679, 45.7389679
32: -36.1053810, 14.9813128, -36.1053810, 14.9813128, -49.6382141, 49.6373444
33: -50.8679428, 21.5127850, -50.8679428, 21.5127850, -69.9116669, 69.9095230
34: -50.5968628, 10.0608883, -50.5968628, 10.0608883, -56.7890167, 56.7858047
35: -42.9593925, 16.9696846, -42.9593925, 16.9696846, -57.5668411, 57.5651474
36: -41.9535332, 18.6489067, -41.9535332, 18.6489067, -60.5203857, 60.5202179
37: -55.7414169, 13.1964712, -55.7414169, 13.1964712, -67.8636169, 67.8623352
38: -52.8991852, 15.9395485, -52.8991852, 15.9395485, -68.8387299, 68.8387299
39: -61.7800980, 18.2122650, -61.7800980, 18.2122650, -79.5843964, 79.5838699
40: -48.4437256, 11.8220787, -48.4437256, 11.8220787, -59.2859344, 59.2850227
41: -35.9136467, 18.6692276, -35.9136467, 18.6692276, -53.4089584, 53.4083557
42: -26.5274906, 13.2349625, -26.5274906, 13.2349625, -38.2336197, 38.2326012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=362, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1574

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 628

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5390531, upper bound: 27.5393372
time: 65.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5358410, upper bound: 27.5425321
time: 60.91 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 128.36 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 128.36
Output dim: 31, lower bound: -27.5454011, upper bound: 27.5391322
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 128.36
Output dim: 31, lower bound: -27.5427677, upper bound: 27.5417687
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 128.36
Output dim: 31, lower bound: -27.5390531, upper bound: 27.5393372
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 128.36
Output dim: 31, lower bound: -27.5358410, upper bound: 27.5425321

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -47.0727310, 16.0515289, -47.0727310, 16.0515289, -63.1242599, 63.1242599
1: -25.2549191, 18.6986504, -25.2549191, 18.6986504, -43.7317429, 43.7315559
2: -21.5161400, 17.9457283, -21.5161400, 17.9457283, -39.4263840, 39.4256287
3: -25.0356750, 17.5100574, -25.0356750, 17.5100574, -42.5457306, 42.5457306
4: -28.3546085, 20.1954041, -28.3546085, 20.1954041, -48.5500107, 48.5500107
5: -23.6489449, 19.8753014, -23.6489449, 19.8753014, -43.5242462, 43.5242462
6: -33.7565460, 18.9100819, -33.7565460, 18.9100819, -51.8763809, 51.8769379
7: -30.3117294, 21.4584789, -30.3117294, 21.4584789, -51.2689590, 51.2682533
8: -32.2651901, 24.0401077, -32.2651901, 24.0401077, -56.3052979, 56.3052979
9: -26.7325325, 17.7273445, -26.7325325, 17.7273445, -44.4598770, 44.4598770
10: -36.5454521, 26.0213966, -36.5454521, 26.0213966, -62.5668488, 62.5668488
11: -27.4929466, 25.1164017, -27.4929466, 25.1164017, -52.6093483, 52.6093483
12: -35.1603775, 22.0571594, -35.1603775, 22.0571594, -55.7988510, 55.8004570
13: -40.0848770, 22.5472832, -40.0848770, 22.5472832, -62.6321602, 62.6321602
14: -56.5082397, 13.5440235, -56.5082397, 13.5440235, -70.0522614, 70.0522614
15: -28.3149033, 19.6139297, -28.3149033, 19.6139297, -47.9288330, 47.9288330
16: -34.9394836, 22.3987980, -34.9394836, 22.3987980, -57.3382797, 57.3382797
17: -51.8795128, 27.0620499, -51.8795128, 27.0620499, -78.9415588, 78.9415588
18: -30.0645676, 26.9229984, -30.0645676, 26.9229984, -56.9875641, 56.9875641
19: -17.7139587, 17.6513634, -17.7139587, 17.6513634, -35.2626038, 35.2619400
20: -20.0517960, 17.4920921, -20.0517960, 17.4920921, -37.5438881, 37.5438881
21: -24.4986229, 22.8401794, -24.4986229, 22.8401794, -47.3388023, 47.3388023
22: -25.5391426, 21.6915550, -25.5391426, 21.6915550, -47.2306976, 47.2306976
23: -18.7391624, 21.6828156, -18.7391624, 21.6828156, -40.4219780, 40.4219780
24: -22.4943600, 22.3042927, -22.4943600, 22.3042927, -44.7986526, 44.7986526
25: -19.4939919, 24.6075783, -19.4939919, 24.6075783, -44.1015701, 44.1015701
26: -33.8213387, 29.5184746, -33.8213387, 29.5184746, -63.3398132, 63.3398132
27: -24.1218414, 22.8650970, -24.1218414, 22.8650970, -46.9869385, 46.9869385
28: -18.7543030, 25.4579201, -18.7543030, 25.4579201, -44.0805969, 44.0803566
29: -25.6286564, 24.3507462, -25.6286564, 24.3507462, -49.9794006, 49.9794006
30: -25.7671089, 25.4406376, -25.7671089, 25.4406376, -51.2077484, 51.2077484
31: -21.9189606, 23.8200073, -21.9189606, 23.8200073, -45.7389679, 45.7389679
32: -36.1053810, 14.9813128, -36.1053810, 14.9813128, -49.6377258, 49.6387100
33: -50.8679428, 21.5127850, -50.8679428, 21.5127850, -69.9080048, 69.9101639
34: -50.5968628, 10.0608883, -50.5968628, 10.0608883, -56.7837143, 56.7869415
35: -42.9593925, 16.9696846, -42.9593925, 16.9696846, -57.5646133, 57.5663071
36: -41.9535332, 18.6489067, -41.9535332, 18.6489067, -60.5200119, 60.5201492
37: -55.7414169, 13.1964712, -55.7414169, 13.1964712, -67.8611298, 67.8624115
38: -52.8991852, 15.9395485, -52.8991852, 15.9395485, -68.8387299, 68.8387299
39: -61.7800980, 18.2122650, -61.7800980, 18.2122650, -79.5838928, 79.5844421
40: -48.4437256, 11.8220787, -48.4437256, 11.8220787, -59.2828903, 59.2838669
41: -35.9136467, 18.6692276, -35.9136467, 18.6692276, -53.4068909, 53.4075203
42: -26.5274906, 13.2349625, -26.5274906, 13.2349625, -38.2341003, 38.2348557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=362, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 717

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 998

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5453799, upper bound: 27.5307089
time: 63.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5369799, upper bound: 27.5391109
time: 60.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -47.0727310, 16.0515289, -47.0727310, 16.0515289, -63.1242599, 63.1242599
1: -25.2549191, 18.6986504, -25.2549191, 18.6986504, -43.7317352, 43.7315559
2: -21.5161400, 17.9457283, -21.5161400, 17.9457283, -39.4263840, 39.4256363
3: -25.0356750, 17.5100574, -25.0356750, 17.5100574, -42.5457306, 42.5457306
4: -28.3546085, 20.1954041, -28.3546085, 20.1954041, -48.5500107, 48.5500107
5: -23.6489449, 19.8753014, -23.6489449, 19.8753014, -43.5242462, 43.5242462
6: -33.7565460, 18.9100819, -33.7565460, 18.9100819, -51.8761826, 51.8771362
7: -30.3117294, 21.4584789, -30.3117294, 21.4584789, -51.2689590, 51.2682571
8: -32.2651901, 24.0401077, -32.2651901, 24.0401077, -56.3052979, 56.3052979
9: -26.7325325, 17.7273445, -26.7325325, 17.7273445, -44.4598770, 44.4598770
10: -36.5454521, 26.0213966, -36.5454521, 26.0213966, -62.5668488, 62.5668488
11: -27.4929466, 25.1164017, -27.4929466, 25.1164017, -52.6093483, 52.6093483
12: -35.1603775, 22.0571594, -35.1603775, 22.0571594, -55.7988815, 55.8004227
13: -40.0848770, 22.5472832, -40.0848770, 22.5472832, -62.6321602, 62.6321602
14: -56.5082397, 13.5440235, -56.5082397, 13.5440235, -70.0522614, 70.0522614
15: -28.3149033, 19.6139297, -28.3149033, 19.6139297, -47.9288330, 47.9288330
16: -34.9394836, 22.3987980, -34.9394836, 22.3987980, -57.3382797, 57.3382797
17: -51.8795128, 27.0620499, -51.8795128, 27.0620499, -78.9415588, 78.9415588
18: -30.0645676, 26.9229984, -30.0645676, 26.9229984, -56.9875641, 56.9875641
19: -17.7139587, 17.6513634, -17.7139587, 17.6513634, -35.2625351, 35.2620087
20: -20.0517960, 17.4920921, -20.0517960, 17.4920921, -37.5438881, 37.5438881
21: -24.4986229, 22.8401794, -24.4986229, 22.8401794, -47.3388023, 47.3388023
22: -25.5391426, 21.6915550, -25.5391426, 21.6915550, -47.2306976, 47.2306976
23: -18.7391624, 21.6828156, -18.7391624, 21.6828156, -40.4219780, 40.4219780
24: -22.4943600, 22.3042927, -22.4943600, 22.3042927, -44.7986526, 44.7986526
25: -19.4939919, 24.6075783, -19.4939919, 24.6075783, -44.1015701, 44.1015701
26: -33.8213387, 29.5184746, -33.8213387, 29.5184746, -63.3398132, 63.3398132
27: -24.1218414, 22.8650970, -24.1218414, 22.8650970, -46.9869385, 46.9869385
28: -18.7543030, 25.4579201, -18.7543030, 25.4579201, -44.0806732, 44.0802803
29: -25.6286564, 24.3507462, -25.6286564, 24.3507462, -49.9794006, 49.9794006
30: -25.7671089, 25.4406376, -25.7671089, 25.4406376, -51.2077484, 51.2077484
31: -21.9189606, 23.8200073, -21.9189606, 23.8200073, -45.7389679, 45.7389679
32: -36.1053810, 14.9813128, -36.1053810, 14.9813128, -49.6378326, 49.6385956
33: -50.8679428, 21.5127850, -50.8679428, 21.5127850, -69.9080200, 69.9101486
34: -50.5968628, 10.0608883, -50.5968628, 10.0608883, -56.7837296, 56.7869263
35: -42.9593925, 16.9696846, -42.9593925, 16.9696846, -57.5646133, 57.5663071
36: -41.9535332, 18.6489067, -41.9535332, 18.6489067, -60.5199661, 60.5201874
37: -55.7414169, 13.1964712, -55.7414169, 13.1964712, -67.8611298, 67.8623962
38: -52.8991852, 15.9395485, -52.8991852, 15.9395485, -68.8387299, 68.8387299
39: -61.7800980, 18.2122650, -61.7800980, 18.2122650, -79.5839233, 79.5844421
40: -48.4437256, 11.8220787, -48.4437256, 11.8220787, -59.2829666, 59.2837906
41: -35.9136467, 18.6692276, -35.9136467, 18.6692276, -53.4069138, 53.4075012
42: -26.5274906, 13.2349625, -26.5274906, 13.2349625, -38.2338409, 38.2351227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=362, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1539

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1639

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5261936, upper bound: 27.5162575
time: 61.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5171648, upper bound: 27.5253322
time: 61.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -47.0727310, 16.0515289, -47.0727310, 16.0515289, -63.1242599, 63.1242599
1: -25.2549191, 18.6986504, -25.2549191, 18.6986504, -43.7311440, 43.7314796
2: -21.5161400, 17.9457283, -21.5161400, 17.9457283, -39.4258690, 39.4267845
3: -25.0356750, 17.5100574, -25.0356750, 17.5100574, -42.5457306, 42.5457306
4: -28.3546085, 20.1954041, -28.3546085, 20.1954041, -48.5500107, 48.5500107
5: -23.6489449, 19.8753014, -23.6489449, 19.8753014, -43.5242462, 43.5242462
6: -33.7565460, 18.9100819, -33.7565460, 18.9100819, -51.8708344, 51.8699417
7: -30.3117294, 21.4584789, -30.3117294, 21.4584789, -51.2681122, 51.2690811
8: -32.2651901, 24.0401077, -32.2651901, 24.0401077, -56.3052979, 56.3052979
9: -26.7325325, 17.7273445, -26.7325325, 17.7273445, -44.4598770, 44.4598770
10: -36.5454521, 26.0213966, -36.5454521, 26.0213966, -62.5668488, 62.5668488
11: -27.4929466, 25.1164017, -27.4929466, 25.1164017, -52.6093483, 52.6093483
12: -35.1603775, 22.0571594, -35.1603775, 22.0571594, -55.8006210, 55.7989502
13: -40.0848770, 22.5472832, -40.0848770, 22.5472832, -62.6321602, 62.6321602
14: -56.5082397, 13.5440235, -56.5082397, 13.5440235, -70.0522614, 70.0522614
15: -28.3149033, 19.6139297, -28.3149033, 19.6139297, -47.9288330, 47.9288330
16: -34.9394836, 22.3987980, -34.9394836, 22.3987980, -57.3382797, 57.3382797
17: -51.8795128, 27.0620499, -51.8795128, 27.0620499, -78.9415588, 78.9415588
18: -30.0645676, 26.9229984, -30.0645676, 26.9229984, -56.9875641, 56.9875641
19: -17.7139587, 17.6513634, -17.7139587, 17.6513634, -35.2641296, 35.2647781
20: -20.0517960, 17.4920921, -20.0517960, 17.4920921, -37.5438881, 37.5438881
21: -24.4986229, 22.8401794, -24.4986229, 22.8401794, -47.3388023, 47.3388023
22: -25.5391426, 21.6915550, -25.5391426, 21.6915550, -47.2306976, 47.2306976
23: -18.7391624, 21.6828156, -18.7391624, 21.6828156, -40.4219780, 40.4219780
24: -22.4943600, 22.3042927, -22.4943600, 22.3042927, -44.7986526, 44.7986526
25: -19.4939919, 24.6075783, -19.4939919, 24.6075783, -44.1015701, 44.1015701
26: -33.8213387, 29.5184746, -33.8213387, 29.5184746, -63.3398132, 63.3398132
27: -24.1218414, 22.8650970, -24.1218414, 22.8650970, -46.9869385, 46.9869385
28: -18.7543030, 25.4579201, -18.7543030, 25.4579201, -44.0818329, 44.0822334
29: -25.6286564, 24.3507462, -25.6286564, 24.3507462, -49.9794006, 49.9794006
30: -25.7671089, 25.4406376, -25.7671089, 25.4406376, -51.2077484, 51.2077484
31: -21.9189606, 23.8200073, -21.9189606, 23.8200073, -45.7389679, 45.7389679
32: -36.1053810, 14.9813128, -36.1053810, 14.9813128, -49.6380768, 49.6371613
33: -50.8679428, 21.5127850, -50.8679428, 21.5127850, -69.9115601, 69.9089050
34: -50.5968628, 10.0608883, -50.5968628, 10.0608883, -56.7888794, 56.7847862
35: -42.9593925, 16.9696846, -42.9593925, 16.9696846, -57.5667343, 57.5643616
36: -41.9535332, 18.6489067, -41.9535332, 18.6489067, -60.5202026, 60.5199814
37: -55.7414169, 13.1964712, -55.7414169, 13.1964712, -67.8635712, 67.8621292
38: -52.8991852, 15.9395485, -52.8991852, 15.9395485, -68.8387299, 68.8387299
39: -61.7800980, 18.2122650, -61.7800980, 18.2122650, -79.5841675, 79.5833435
40: -48.4437256, 11.8220787, -48.4437256, 11.8220787, -59.2859344, 59.2850113
41: -35.9136467, 18.6692276, -35.9136467, 18.6692276, -53.4086456, 53.4079552
42: -26.5274906, 13.2349625, -26.5274906, 13.2349625, -38.2316589, 38.2304192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=362, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 870

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1686

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5312453, upper bound: 27.5314242
time: 80.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5311453, upper bound: 27.5315139
time: 65.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -47.0727310, 16.0515289, -47.0727310, 16.0515289, -63.1242599, 63.1242599
1: -25.2549191, 18.6986504, -25.2549191, 18.6986504, -43.7312965, 43.7313271
2: -21.5161400, 17.9457283, -21.5161400, 17.9457283, -39.4260368, 39.4266205
3: -25.0356750, 17.5100574, -25.0356750, 17.5100574, -42.5457306, 42.5457306
4: -28.3546085, 20.1954041, -28.3546085, 20.1954041, -48.5500107, 48.5500107
5: -23.6489449, 19.8753014, -23.6489449, 19.8753014, -43.5242462, 43.5242462
6: -33.7565460, 18.9100819, -33.7565460, 18.9100819, -51.8706970, 51.8700790
7: -30.3117294, 21.4584789, -30.3117294, 21.4584789, -51.2683868, 51.2688103
8: -32.2651901, 24.0401077, -32.2651901, 24.0401077, -56.3052979, 56.3052979
9: -26.7325325, 17.7273445, -26.7325325, 17.7273445, -44.4598770, 44.4598770
10: -36.5454521, 26.0213966, -36.5454521, 26.0213966, -62.5668488, 62.5668488
11: -27.4929466, 25.1164017, -27.4929466, 25.1164017, -52.6093483, 52.6093483
12: -35.1603775, 22.0571594, -35.1603775, 22.0571594, -55.8005295, 55.7990532
13: -40.0848770, 22.5472832, -40.0848770, 22.5472832, -62.6321602, 62.6321602
14: -56.5082397, 13.5440235, -56.5082397, 13.5440235, -70.0522614, 70.0522614
15: -28.3149033, 19.6139297, -28.3149033, 19.6139297, -47.9288330, 47.9288330
16: -34.9394836, 22.3987980, -34.9394836, 22.3987980, -57.3382797, 57.3382797
17: -51.8795128, 27.0620499, -51.8795128, 27.0620499, -78.9415588, 78.9415588
18: -30.0645676, 26.9229984, -30.0645676, 26.9229984, -56.9875641, 56.9875641
19: -17.7139587, 17.6513634, -17.7139587, 17.6513634, -35.2641830, 35.2647247
20: -20.0517960, 17.4920921, -20.0517960, 17.4920921, -37.5438881, 37.5438881
21: -24.4986229, 22.8401794, -24.4986229, 22.8401794, -47.3388023, 47.3388023
22: -25.5391426, 21.6915550, -25.5391426, 21.6915550, -47.2306976, 47.2306976
23: -18.7391624, 21.6828156, -18.7391624, 21.6828156, -40.4219780, 40.4219780
24: -22.4943600, 22.3042927, -22.4943600, 22.3042927, -44.7986526, 44.7986526
25: -19.4939919, 24.6075783, -19.4939919, 24.6075783, -44.1015701, 44.1015701
26: -33.8213387, 29.5184746, -33.8213387, 29.5184746, -63.3398132, 63.3398132
27: -24.1218414, 22.8650970, -24.1218414, 22.8650970, -46.9869385, 46.9869385
28: -18.7543030, 25.4579201, -18.7543030, 25.4579201, -44.0819168, 44.0821495
29: -25.6286564, 24.3507462, -25.6286564, 24.3507462, -49.9794006, 49.9794006
30: -25.7671089, 25.4406376, -25.7671089, 25.4406376, -51.2077484, 51.2077484
31: -21.9189606, 23.8200073, -21.9189606, 23.8200073, -45.7389679, 45.7389679
32: -36.1053810, 14.9813128, -36.1053810, 14.9813128, -49.6380310, 49.6372032
33: -50.8679428, 21.5127850, -50.8679428, 21.5127850, -69.9110413, 69.9094238
34: -50.5968628, 10.0608883, -50.5968628, 10.0608883, -56.7880020, 56.7856674
35: -42.9593925, 16.9696846, -42.9593925, 16.9696846, -57.5660629, 57.5650368
36: -41.9535332, 18.6489067, -41.9535332, 18.6489067, -60.5201569, 60.5200272
37: -55.7414169, 13.1964712, -55.7414169, 13.1964712, -67.8634033, 67.8623047
38: -52.8991852, 15.9395485, -52.8991852, 15.9395485, -68.8387299, 68.8387299
39: -61.7800980, 18.2122650, -61.7800980, 18.2122650, -79.5838776, 79.5836258
40: -48.4437256, 11.8220787, -48.4437256, 11.8220787, -59.2859192, 59.2850266
41: -35.9136467, 18.6692276, -35.9136467, 18.6692276, -53.4085617, 53.4080353
42: -26.5274906, 13.2349625, -26.5274906, 13.2349625, -38.2314377, 38.2306442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=362, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1664

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 997

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5356663, upper bound: 27.5411397
time: 61.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5344481, upper bound: 27.5423571
time: 60.95 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 124.16 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 124.16
Output dim: 31, lower bound: -27.5453799, upper bound: 27.5307089
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 124.16
Output dim: 31, lower bound: -27.5369799, upper bound: 27.5391109
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 124.16
Output dim: 31, lower bound: -27.5261936, upper bound: 27.5162575
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 124.16
Output dim: 31, lower bound: -27.5171648, upper bound: 27.5253322
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 124.16
Output dim: 31, lower bound: -27.5312453, upper bound: 27.5314242
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 124.16
Output dim: 31, lower bound: -27.5311453, upper bound: 27.5315139
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 124.16
Output dim: 31, lower bound: -27.5356663, upper bound: 27.5411397
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 124.16
Output dim: 31, lower bound: -27.5344481, upper bound: 27.5423571

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -47.0727310, 16.0515289, -47.0727310, 16.0515289, -63.1242599, 63.1242599
1: -25.2549191, 18.6986504, -25.2549191, 18.6986504, -43.7312469, 43.7311974
2: -21.5161400, 17.9457283, -21.5161400, 17.9457283, -39.4256668, 39.4251099
3: -25.0356750, 17.5100574, -25.0356750, 17.5100574, -42.5457306, 42.5457306
4: -28.3546085, 20.1954041, -28.3546085, 20.1954041, -48.5500107, 48.5500107
5: -23.6489449, 19.8753014, -23.6489449, 19.8753014, -43.5242462, 43.5242462
6: -33.7565460, 18.9100819, -33.7565460, 18.9100819, -51.8761902, 51.8767586
7: -30.3117294, 21.4584789, -30.3117294, 21.4584789, -51.2679825, 51.2675323
8: -32.2651901, 24.0401077, -32.2651901, 24.0401077, -56.3052979, 56.3052979
9: -26.7325325, 17.7273445, -26.7325325, 17.7273445, -44.4598770, 44.4598770
10: -36.5454521, 26.0213966, -36.5454521, 26.0213966, -62.5668488, 62.5668488
11: -27.4929466, 25.1164017, -27.4929466, 25.1164017, -52.6093483, 52.6093483
12: -35.1603775, 22.0571594, -35.1603775, 22.0571594, -55.7981949, 55.7995453
13: -40.0848770, 22.5472832, -40.0848770, 22.5472832, -62.6321602, 62.6321602
14: -56.5082397, 13.5440235, -56.5082397, 13.5440235, -70.0522614, 70.0522614
15: -28.3149033, 19.6139297, -28.3149033, 19.6139297, -47.9288330, 47.9288330
16: -34.9394836, 22.3987980, -34.9394836, 22.3987980, -57.3382797, 57.3382797
17: -51.8795128, 27.0620499, -51.8795128, 27.0620499, -78.9415588, 78.9415588
18: -30.0645676, 26.9229984, -30.0645676, 26.9229984, -56.9875641, 56.9875641
19: -17.7139587, 17.6513634, -17.7139587, 17.6513634, -35.2625809, 35.2619553
20: -20.0517960, 17.4920921, -20.0517960, 17.4920921, -37.5438881, 37.5438881
21: -24.4986229, 22.8401794, -24.4986229, 22.8401794, -47.3388023, 47.3388023
22: -25.5391426, 21.6915550, -25.5391426, 21.6915550, -47.2306976, 47.2306976
23: -18.7391624, 21.6828156, -18.7391624, 21.6828156, -40.4219780, 40.4219780
24: -22.4943600, 22.3042927, -22.4943600, 22.3042927, -44.7986526, 44.7986526
25: -19.4939919, 24.6075783, -19.4939919, 24.6075783, -44.1015701, 44.1015701
26: -33.8213387, 29.5184746, -33.8213387, 29.5184746, -63.3398132, 63.3398132
27: -24.1218414, 22.8650970, -24.1218414, 22.8650970, -46.9869385, 46.9869385
28: -18.7543030, 25.4579201, -18.7543030, 25.4579201, -44.0804787, 44.0802536
29: -25.6286564, 24.3507462, -25.6286564, 24.3507462, -49.9794006, 49.9794006
30: -25.7671089, 25.4406376, -25.7671089, 25.4406376, -51.2077484, 51.2077484
31: -21.9189606, 23.8200073, -21.9189606, 23.8200073, -45.7389679, 45.7389679
32: -36.1053810, 14.9813128, -36.1053810, 14.9813128, -49.6375427, 49.6384506
33: -50.8679428, 21.5127850, -50.8679428, 21.5127850, -69.9066925, 69.9082794
34: -50.5968628, 10.0608883, -50.5968628, 10.0608883, -56.7818565, 56.7841873
35: -42.9593925, 16.9696846, -42.9593925, 16.9696846, -57.5628433, 57.5639191
36: -41.9535332, 18.6489067, -41.9535332, 18.6489067, -60.5199966, 60.5201263
37: -55.7414169, 13.1964712, -55.7414169, 13.1964712, -67.8604279, 67.8614426
38: -52.8991852, 15.9395485, -52.8991852, 15.9395485, -68.8387299, 68.8387299
39: -61.7800980, 18.2122650, -61.7800980, 18.2122650, -79.5835724, 79.5838547
40: -48.4437256, 11.8220787, -48.4437256, 11.8220787, -59.2825775, 59.2834435
41: -35.9136467, 18.6692276, -35.9136467, 18.6692276, -53.4068604, 53.4074707
42: -26.5274906, 13.2349625, -26.5274906, 13.2349625, -38.2334595, 38.2342072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=362, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 757

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 870

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5413301, upper bound: 27.5305294
time: 63.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5452282, upper bound: 27.5269178
time: 63.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -47.0727310, 16.0515289, -47.0727310, 16.0515289, -63.1242599, 63.1242599
1: -25.2549191, 18.6986504, -25.2549191, 18.6986504, -43.7313843, 43.7310600
2: -21.5161400, 17.9457283, -21.5161400, 17.9457283, -39.4258652, 39.4249153
3: -25.0356750, 17.5100574, -25.0356750, 17.5100574, -42.5457306, 42.5457306
4: -28.3546085, 20.1954041, -28.3546085, 20.1954041, -48.5500107, 48.5500107
5: -23.6489449, 19.8753014, -23.6489449, 19.8753014, -43.5242462, 43.5242462
6: -33.7565460, 18.9100819, -33.7565460, 18.9100819, -51.8761978, 51.8767471
7: -30.3117294, 21.4584789, -30.3117294, 21.4584789, -51.2682266, 51.2672806
8: -32.2651901, 24.0401077, -32.2651901, 24.0401077, -56.3052979, 56.3052979
9: -26.7325325, 17.7273445, -26.7325325, 17.7273445, -44.4598770, 44.4598770
10: -36.5454521, 26.0213966, -36.5454521, 26.0213966, -62.5668488, 62.5668488
11: -27.4929466, 25.1164017, -27.4929466, 25.1164017, -52.6093483, 52.6093483
12: -35.1603775, 22.0571594, -35.1603775, 22.0571594, -55.7979507, 55.7997971
13: -40.0848770, 22.5472832, -40.0848770, 22.5472832, -62.6321602, 62.6321602
14: -56.5082397, 13.5440235, -56.5082397, 13.5440235, -70.0522614, 70.0522614
15: -28.3149033, 19.6139297, -28.3149033, 19.6139297, -47.9288330, 47.9288330
16: -34.9394836, 22.3987980, -34.9394836, 22.3987980, -57.3382797, 57.3382797
17: -51.8795128, 27.0620499, -51.8795128, 27.0620499, -78.9415588, 78.9415588
18: -30.0645676, 26.9229984, -30.0645676, 26.9229984, -56.9875641, 56.9875641
19: -17.7139587, 17.6513634, -17.7139587, 17.6513634, -35.2626266, 35.2619209
20: -20.0517960, 17.4920921, -20.0517960, 17.4920921, -37.5438881, 37.5438881
21: -24.4986229, 22.8401794, -24.4986229, 22.8401794, -47.3388023, 47.3388023
22: -25.5391426, 21.6915550, -25.5391426, 21.6915550, -47.2306976, 47.2306976
23: -18.7391624, 21.6828156, -18.7391624, 21.6828156, -40.4219780, 40.4219780
24: -22.4943600, 22.3042927, -22.4943600, 22.3042927, -44.7986526, 44.7986526
25: -19.4939919, 24.6075783, -19.4939919, 24.6075783, -44.1015701, 44.1015701
26: -33.8213387, 29.5184746, -33.8213387, 29.5184746, -63.3398132, 63.3398132
27: -24.1218414, 22.8650970, -24.1218414, 22.8650970, -46.9869385, 46.9869385
28: -18.7543030, 25.4579201, -18.7543030, 25.4579201, -44.0804939, 44.0802383
29: -25.6286564, 24.3507462, -25.6286564, 24.3507462, -49.9794006, 49.9794006
30: -25.7671089, 25.4406376, -25.7671089, 25.4406376, -51.2077484, 51.2077484
31: -21.9189606, 23.8200073, -21.9189606, 23.8200073, -45.7389679, 45.7389679
32: -36.1053810, 14.9813128, -36.1053810, 14.9813128, -49.6374664, 49.6385231
33: -50.8679428, 21.5127850, -50.8679428, 21.5127850, -69.9061279, 69.9088516
34: -50.5968628, 10.0608883, -50.5968628, 10.0608883, -56.7809639, 56.7850800
35: -42.9593925, 16.9696846, -42.9593925, 16.9696846, -57.5622253, 57.5645409
36: -41.9535332, 18.6489067, -41.9535332, 18.6489067, -60.5199966, 60.5201378
37: -55.7414169, 13.1964712, -55.7414169, 13.1964712, -67.8601685, 67.8617172
38: -52.8991852, 15.9395485, -52.8991852, 15.9395485, -68.8387299, 68.8387299
39: -61.7800980, 18.2122650, -61.7800980, 18.2122650, -79.5833282, 79.5840988
40: -48.4437256, 11.8220787, -48.4437256, 11.8220787, -59.2824554, 59.2835503
41: -35.9136467, 18.6692276, -35.9136467, 18.6692276, -53.4068451, 53.4074936
42: -26.5274906, 13.2349625, -26.5274906, 13.2349625, -38.2334595, 38.2342148

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=362, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1000

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 755

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5320488, upper bound: 27.5389103
time: 64.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5367801, upper bound: 27.5341836
time: 54.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -47.0727310, 16.0515289, -47.0727310, 16.0515289, -63.1242599, 63.1242599
1: -25.2549191, 18.6986504, -25.2549191, 18.6986504, -43.7317314, 43.7315483
2: -21.5161400, 17.9457283, -21.5161400, 17.9457283, -39.4263115, 39.4254837
3: -25.0356750, 17.5100574, -25.0356750, 17.5100574, -42.5457306, 42.5457306
4: -28.3546085, 20.1954041, -28.3546085, 20.1954041, -48.5500107, 48.5500107
5: -23.6489449, 19.8753014, -23.6489449, 19.8753014, -43.5242462, 43.5242462
6: -33.7565460, 18.9100819, -33.7565460, 18.9100819, -51.8749466, 51.8765182
7: -30.3117294, 21.4584789, -30.3117294, 21.4584789, -51.2688522, 51.2680969
8: -32.2651901, 24.0401077, -32.2651901, 24.0401077, -56.3052979, 56.3052979
9: -26.7325325, 17.7273445, -26.7325325, 17.7273445, -44.4598770, 44.4598770
10: -36.5454521, 26.0213966, -36.5454521, 26.0213966, -62.5668488, 62.5668488
11: -27.4929466, 25.1164017, -27.4929466, 25.1164017, -52.6093483, 52.6093483
12: -35.1603775, 22.0571594, -35.1603775, 22.0571594, -55.7983170, 55.8001175
13: -40.0848770, 22.5472832, -40.0848770, 22.5472832, -62.6321602, 62.6321602
14: -56.5082397, 13.5440235, -56.5082397, 13.5440235, -70.0522614, 70.0522614
15: -28.3149033, 19.6139297, -28.3149033, 19.6139297, -47.9288330, 47.9288330
16: -34.9394836, 22.3987980, -34.9394836, 22.3987980, -57.3382797, 57.3382797
17: -51.8795128, 27.0620499, -51.8795128, 27.0620499, -78.9415588, 78.9415588
18: -30.0645676, 26.9229984, -30.0645676, 26.9229984, -56.9875641, 56.9875641
19: -17.7139587, 17.6513634, -17.7139587, 17.6513634, -35.2620735, 35.2613564
20: -20.0517960, 17.4920921, -20.0517960, 17.4920921, -37.5438881, 37.5438881
21: -24.4986229, 22.8401794, -24.4986229, 22.8401794, -47.3388023, 47.3388023
22: -25.5391426, 21.6915550, -25.5391426, 21.6915550, -47.2306976, 47.2306976
23: -18.7391624, 21.6828156, -18.7391624, 21.6828156, -40.4219780, 40.4219780
24: -22.4943600, 22.3042927, -22.4943600, 22.3042927, -44.7986526, 44.7986526
25: -19.4939919, 24.6075783, -19.4939919, 24.6075783, -44.1015701, 44.1015701
26: -33.8213387, 29.5184746, -33.8213387, 29.5184746, -63.3398132, 63.3398132
27: -24.1218414, 22.8650970, -24.1218414, 22.8650970, -46.9869385, 46.9869385
28: -18.7543030, 25.4579201, -18.7543030, 25.4579201, -44.0804329, 44.0798569
29: -25.6286564, 24.3507462, -25.6286564, 24.3507462, -49.9794006, 49.9794006
30: -25.7671089, 25.4406376, -25.7671089, 25.4406376, -51.2077484, 51.2077484
31: -21.9189606, 23.8200073, -21.9189606, 23.8200073, -45.7389679, 45.7389679
32: -36.1053810, 14.9813128, -36.1053810, 14.9813128, -49.6370392, 49.6380348
33: -50.8679428, 21.5127850, -50.8679428, 21.5127850, -69.9073486, 69.9096909
34: -50.5968628, 10.0608883, -50.5968628, 10.0608883, -56.7828674, 56.7863693
35: -42.9593925, 16.9696846, -42.9593925, 16.9696846, -57.5641785, 57.5660019
36: -41.9535332, 18.6489067, -41.9535332, 18.6489067, -60.5197067, 60.5200195
37: -55.7414169, 13.1964712, -55.7414169, 13.1964712, -67.8608551, 67.8622742
38: -52.8991852, 15.9395485, -52.8991852, 15.9395485, -68.8387299, 68.8387299
39: -61.7800980, 18.2122650, -61.7800980, 18.2122650, -79.5838013, 79.5843353
40: -48.4437256, 11.8220787, -48.4437256, 11.8220787, -59.2823868, 59.2834015
41: -35.9136467, 18.6692276, -35.9136467, 18.6692276, -53.4061584, 53.4069672
42: -26.5274906, 13.2349625, -26.5274906, 13.2349625, -38.2324066, 38.2345161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=362, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 624

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 750

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5245350, upper bound: 27.5161947
time: 51.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5261307, upper bound: 27.5146068
time: 80.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -47.0727310, 16.0515289, -47.0727310, 16.0515289, -63.1242599, 63.1242599
1: -25.2549191, 18.6986504, -25.2549191, 18.6986504, -43.7317314, 43.7315483
2: -21.5161400, 17.9457283, -21.5161400, 17.9457283, -39.4262199, 39.4255676
3: -25.0356750, 17.5100574, -25.0356750, 17.5100574, -42.5457306, 42.5457306
4: -28.3546085, 20.1954041, -28.3546085, 20.1954041, -48.5500107, 48.5500107
5: -23.6489449, 19.8753014, -23.6489449, 19.8753014, -43.5242462, 43.5242462
6: -33.7565460, 18.9100819, -33.7565460, 18.9100819, -51.8755646, 51.8758965
7: -30.3117294, 21.4584789, -30.3117294, 21.4584789, -51.2687912, 51.2681541
8: -32.2651901, 24.0401077, -32.2651901, 24.0401077, -56.3052979, 56.3052979
9: -26.7325325, 17.7273445, -26.7325325, 17.7273445, -44.4598770, 44.4598770
10: -36.5454521, 26.0213966, -36.5454521, 26.0213966, -62.5668488, 62.5668488
11: -27.4929466, 25.1164017, -27.4929466, 25.1164017, -52.6093483, 52.6093483
12: -35.1603775, 22.0571594, -35.1603775, 22.0571594, -55.7985916, 55.7998505
13: -40.0848770, 22.5472832, -40.0848770, 22.5472832, -62.6321602, 62.6321602
14: -56.5082397, 13.5440235, -56.5082397, 13.5440235, -70.0522614, 70.0522614
15: -28.3149033, 19.6139297, -28.3149033, 19.6139297, -47.9288330, 47.9288330
16: -34.9394836, 22.3987980, -34.9394836, 22.3987980, -57.3382797, 57.3382797
17: -51.8795128, 27.0620499, -51.8795128, 27.0620499, -78.9415588, 78.9415588
18: -30.0645676, 26.9229984, -30.0645676, 26.9229984, -56.9875641, 56.9875641
19: -17.7139587, 17.6513634, -17.7139587, 17.6513634, -35.2618752, 35.2615509
20: -20.0517960, 17.4920921, -20.0517960, 17.4920921, -37.5438881, 37.5438881
21: -24.4986229, 22.8401794, -24.4986229, 22.8401794, -47.3388023, 47.3388023
22: -25.5391426, 21.6915550, -25.5391426, 21.6915550, -47.2306976, 47.2306976
23: -18.7391624, 21.6828156, -18.7391624, 21.6828156, -40.4219780, 40.4219780
24: -22.4943600, 22.3042927, -22.4943600, 22.3042927, -44.7986526, 44.7986526
25: -19.4939919, 24.6075783, -19.4939919, 24.6075783, -44.1015701, 44.1015701
26: -33.8213387, 29.5184746, -33.8213387, 29.5184746, -63.3398132, 63.3398132
27: -24.1218414, 22.8650970, -24.1218414, 22.8650970, -46.9869385, 46.9869385
28: -18.7543030, 25.4579201, -18.7543030, 25.4579201, -44.0802498, 44.0800438
29: -25.6286564, 24.3507462, -25.6286564, 24.3507462, -49.9794006, 49.9794006
30: -25.7671089, 25.4406376, -25.7671089, 25.4406376, -51.2077484, 51.2077484
31: -21.9189606, 23.8200073, -21.9189606, 23.8200073, -45.7389679, 45.7389679
32: -36.1053810, 14.9813128, -36.1053810, 14.9813128, -49.6372757, 49.6377945
33: -50.8679428, 21.5127850, -50.8679428, 21.5127850, -69.9075623, 69.9094696
34: -50.5968628, 10.0608883, -50.5968628, 10.0608883, -56.7831726, 56.7860641
35: -42.9593925, 16.9696846, -42.9593925, 16.9696846, -57.5643082, 57.5658722
36: -41.9535332, 18.6489067, -41.9535332, 18.6489067, -60.5198135, 60.5199203
37: -55.7414169, 13.1964712, -55.7414169, 13.1964712, -67.8610077, 67.8621140
38: -52.8991852, 15.9395485, -52.8991852, 15.9395485, -68.8387299, 68.8387299
39: -61.7800980, 18.2122650, -61.7800980, 18.2122650, -79.5838013, 79.5843277
40: -48.4437256, 11.8220787, -48.4437256, 11.8220787, -59.2825699, 59.2832108
41: -35.9136467, 18.6692276, -35.9136467, 18.6692276, -53.4063797, 53.4067383
42: -26.5274906, 13.2349625, -26.5274906, 13.2349625, -38.2332306, 38.2336960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=362, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 750

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 996

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5141754, upper bound: 27.5250751
time: 80.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5169085, upper bound: 27.5223451
time: 60.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -47.0727310, 16.0515289, -47.0727310, 16.0515289, -63.1242599, 63.1242599
1: -25.2549191, 18.6986504, -25.2549191, 18.6986504, -43.7312698, 43.7314262
2: -21.5161400, 17.9457283, -21.5161400, 17.9457283, -39.4258385, 39.4266853
3: -25.0356750, 17.5100574, -25.0356750, 17.5100574, -42.5457306, 42.5457306
4: -28.3546085, 20.1954041, -28.3546085, 20.1954041, -48.5500107, 48.5500107
5: -23.6489449, 19.8753014, -23.6489449, 19.8753014, -43.5242462, 43.5242462
6: -33.7565460, 18.9100819, -33.7565460, 18.9100819, -51.8712006, 51.8697433
7: -30.3117294, 21.4584789, -30.3117294, 21.4584789, -51.2681198, 51.2690620
8: -32.2651901, 24.0401077, -32.2651901, 24.0401077, -56.3052979, 56.3052979
9: -26.7325325, 17.7273445, -26.7325325, 17.7273445, -44.4598770, 44.4598770
10: -36.5454521, 26.0213966, -36.5454521, 26.0213966, -62.5668488, 62.5668488
11: -27.4929466, 25.1164017, -27.4929466, 25.1164017, -52.6093483, 52.6093483
12: -35.1603775, 22.0571594, -35.1603775, 22.0571594, -55.8006744, 55.7986832
13: -40.0848770, 22.5472832, -40.0848770, 22.5472832, -62.6321602, 62.6321602
14: -56.5082397, 13.5440235, -56.5082397, 13.5440235, -70.0522614, 70.0522614
15: -28.3149033, 19.6139297, -28.3149033, 19.6139297, -47.9288330, 47.9288330
16: -34.9394836, 22.3987980, -34.9394836, 22.3987980, -57.3382797, 57.3382797
17: -51.8795128, 27.0620499, -51.8795128, 27.0620499, -78.9415588, 78.9415588
18: -30.0645676, 26.9229984, -30.0645676, 26.9229984, -56.9875641, 56.9875641
19: -17.7139587, 17.6513634, -17.7139587, 17.6513634, -35.2640762, 35.2649460
20: -20.0517960, 17.4920921, -20.0517960, 17.4920921, -37.5438881, 37.5438881
21: -24.4986229, 22.8401794, -24.4986229, 22.8401794, -47.3388023, 47.3388023
22: -25.5391426, 21.6915550, -25.5391426, 21.6915550, -47.2306976, 47.2306976
23: -18.7391624, 21.6828156, -18.7391624, 21.6828156, -40.4219780, 40.4219780
24: -22.4943600, 22.3042927, -22.4943600, 22.3042927, -44.7986526, 44.7986526
25: -19.4939919, 24.6075783, -19.4939919, 24.6075783, -44.1015701, 44.1015701
26: -33.8213387, 29.5184746, -33.8213387, 29.5184746, -63.3398132, 63.3398132
27: -24.1218414, 22.8650970, -24.1218414, 22.8650970, -46.9869385, 46.9869385
28: -18.7543030, 25.4579201, -18.7543030, 25.4579201, -44.0816116, 44.0822029
29: -25.6286564, 24.3507462, -25.6286564, 24.3507462, -49.9794006, 49.9794006
30: -25.7671089, 25.4406376, -25.7671089, 25.4406376, -51.2077484, 51.2077484
31: -21.9189606, 23.8200073, -21.9189606, 23.8200073, -45.7389679, 45.7389679
32: -36.1053810, 14.9813128, -36.1053810, 14.9813128, -49.6382141, 49.6370316
33: -50.8679428, 21.5127850, -50.8679428, 21.5127850, -69.9113617, 69.9089050
34: -50.5968628, 10.0608883, -50.5968628, 10.0608883, -56.7886429, 56.7847748
35: -42.9593925, 16.9696846, -42.9593925, 16.9696846, -57.5667038, 57.5644073
36: -41.9535332, 18.6489067, -41.9535332, 18.6489067, -60.5202789, 60.5199547
37: -55.7414169, 13.1964712, -55.7414169, 13.1964712, -67.8636322, 67.8621140
38: -52.8991852, 15.9395485, -52.8991852, 15.9395485, -68.8387299, 68.8387299
39: -61.7800980, 18.2122650, -61.7800980, 18.2122650, -79.5840912, 79.5834045
40: -48.4437256, 11.8220787, -48.4437256, 11.8220787, -59.2861710, 59.2849960
41: -35.9136467, 18.6692276, -35.9136467, 18.6692276, -53.4088974, 53.4079361
42: -26.5274906, 13.2349625, -26.5274906, 13.2349625, -38.2317619, 38.2297935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=362, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 550

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 533

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5312431, upper bound: 27.5299650
time: 66.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5297939, upper bound: 27.5314221
time: 65.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -47.0727310, 16.0515289, -47.0727310, 16.0515289, -63.1242599, 63.1242599
1: -25.2549191, 18.6986504, -25.2549191, 18.6986504, -43.7310867, 43.7314796
2: -21.5161400, 17.9457283, -21.5161400, 17.9457283, -39.4258690, 39.4267502
3: -25.0356750, 17.5100574, -25.0356750, 17.5100574, -42.5457306, 42.5457306
4: -28.3546085, 20.1954041, -28.3546085, 20.1954041, -48.5500107, 48.5500107
5: -23.6489449, 19.8753014, -23.6489449, 19.8753014, -43.5242462, 43.5242462
6: -33.7565460, 18.9100819, -33.7565460, 18.9100819, -51.8706360, 51.8699417
7: -30.3117294, 21.4584789, -30.3117294, 21.4584789, -51.2680893, 51.2690811
8: -32.2651901, 24.0401077, -32.2651901, 24.0401077, -56.3052979, 56.3052979
9: -26.7325325, 17.7273445, -26.7325325, 17.7273445, -44.4598770, 44.4598770
10: -36.5454521, 26.0213966, -36.5454521, 26.0213966, -62.5668488, 62.5668488
11: -27.4929466, 25.1164017, -27.4929466, 25.1164017, -52.6093483, 52.6093483
12: -35.1603775, 22.0571594, -35.1603775, 22.0571594, -55.8003540, 55.7989502
13: -40.0848770, 22.5472832, -40.0848770, 22.5472832, -62.6321602, 62.6321602
14: -56.5082397, 13.5440235, -56.5082397, 13.5440235, -70.0522614, 70.0522614
15: -28.3149033, 19.6139297, -28.3149033, 19.6139297, -47.9288330, 47.9288330
16: -34.9394836, 22.3987980, -34.9394836, 22.3987980, -57.3382797, 57.3382797
17: -51.8795128, 27.0620499, -51.8795128, 27.0620499, -78.9415588, 78.9415588
18: -30.0645676, 26.9229984, -30.0645676, 26.9229984, -56.9875641, 56.9875641
19: -17.7139587, 17.6513634, -17.7139587, 17.6513634, -35.2641296, 35.2647171
20: -20.0517960, 17.4920921, -20.0517960, 17.4920921, -37.5438881, 37.5438881
21: -24.4986229, 22.8401794, -24.4986229, 22.8401794, -47.3388023, 47.3388023
22: -25.5391426, 21.6915550, -25.5391426, 21.6915550, -47.2306976, 47.2306976
23: -18.7391624, 21.6828156, -18.7391624, 21.6828156, -40.4219780, 40.4219780
24: -22.4943600, 22.3042927, -22.4943600, 22.3042927, -44.7986526, 44.7986526
25: -19.4939919, 24.6075783, -19.4939919, 24.6075783, -44.1015701, 44.1015701
26: -33.8213387, 29.5184746, -33.8213387, 29.5184746, -63.3398132, 63.3398132
27: -24.1218414, 22.8650970, -24.1218414, 22.8650970, -46.9869385, 46.9869385
28: -18.7543030, 25.4579201, -18.7543030, 25.4579201, -44.0818329, 44.0820045
29: -25.6286564, 24.3507462, -25.6286564, 24.3507462, -49.9794006, 49.9794006
30: -25.7671089, 25.4406376, -25.7671089, 25.4406376, -51.2077484, 51.2077484
31: -21.9189606, 23.8200073, -21.9189606, 23.8200073, -45.7389679, 45.7389679
32: -36.1053810, 14.9813128, -36.1053810, 14.9813128, -49.6379471, 49.6371613
33: -50.8679428, 21.5127850, -50.8679428, 21.5127850, -69.9115753, 69.9089050
34: -50.5968628, 10.0608883, -50.5968628, 10.0608883, -56.7888641, 56.7847862
35: -42.9593925, 16.9696846, -42.9593925, 16.9696846, -57.5667343, 57.5643387
36: -41.9535332, 18.6489067, -41.9535332, 18.6489067, -60.5201721, 60.5199814
37: -55.7414169, 13.1964712, -55.7414169, 13.1964712, -67.8635559, 67.8621292
38: -52.8991852, 15.9395485, -52.8991852, 15.9395485, -68.8387299, 68.8387299
39: -61.7800980, 18.2122650, -61.7800980, 18.2122650, -79.5841675, 79.5832672
40: -48.4437256, 11.8220787, -48.4437256, 11.8220787, -59.2859116, 59.2850113
41: -35.9136467, 18.6692276, -35.9136467, 18.6692276, -53.4086227, 53.4079552
42: -26.5274906, 13.2349625, -26.5274906, 13.2349625, -38.2310295, 38.2304192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=362, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 522

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 855

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5149610, upper bound: 27.5313155
time: 64.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5309546, upper bound: 27.5153316
time: 75.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -47.0727310, 16.0515289, -47.0727310, 16.0515289, -63.1242599, 63.1242599
1: -25.2549191, 18.6986504, -25.2549191, 18.6986504, -43.7308998, 43.7310753
2: -21.5161400, 17.9457283, -21.5161400, 17.9457283, -39.4255486, 39.4263382
3: -25.0356750, 17.5100574, -25.0356750, 17.5100574, -42.5457306, 42.5457306
4: -28.3546085, 20.1954041, -28.3546085, 20.1954041, -48.5500107, 48.5500107
5: -23.6489449, 19.8753014, -23.6489449, 19.8753014, -43.5242462, 43.5242462
6: -33.7565460, 18.9100819, -33.7565460, 18.9100819, -51.8701553, 51.8695450
7: -30.3117294, 21.4584789, -30.3117294, 21.4584789, -51.2675476, 51.2682533
8: -32.2651901, 24.0401077, -32.2651901, 24.0401077, -56.3052979, 56.3052979
9: -26.7325325, 17.7273445, -26.7325325, 17.7273445, -44.4598770, 44.4598770
10: -36.5454521, 26.0213966, -36.5454521, 26.0213966, -62.5668488, 62.5668488
11: -27.4929466, 25.1164017, -27.4929466, 25.1164017, -52.6093483, 52.6093483
12: -35.1603775, 22.0571594, -35.1603775, 22.0571594, -55.7997284, 55.7980309
13: -40.0848770, 22.5472832, -40.0848770, 22.5472832, -62.6321602, 62.6321602
14: -56.5082397, 13.5440235, -56.5082397, 13.5440235, -70.0522614, 70.0522614
15: -28.3149033, 19.6139297, -28.3149033, 19.6139297, -47.9288330, 47.9288330
16: -34.9394836, 22.3987980, -34.9394836, 22.3987980, -57.3382797, 57.3382797
17: -51.8795128, 27.0620499, -51.8795128, 27.0620499, -78.9415588, 78.9415588
18: -30.0645676, 26.9229984, -30.0645676, 26.9229984, -56.9875641, 56.9875641
19: -17.7139587, 17.6513634, -17.7139587, 17.6513634, -35.2640610, 35.2646408
20: -20.0517960, 17.4920921, -20.0517960, 17.4920921, -37.5438881, 37.5438881
21: -24.4986229, 22.8401794, -24.4986229, 22.8401794, -47.3388023, 47.3388023
22: -25.5391426, 21.6915550, -25.5391426, 21.6915550, -47.2306976, 47.2306976
23: -18.7391624, 21.6828156, -18.7391624, 21.6828156, -40.4219780, 40.4219780
24: -22.4943600, 22.3042927, -22.4943600, 22.3042927, -44.7986526, 44.7986526
25: -19.4939919, 24.6075783, -19.4939919, 24.6075783, -44.1015701, 44.1015701
26: -33.8213387, 29.5184746, -33.8213387, 29.5184746, -63.3398132, 63.3398132
27: -24.1218414, 22.8650970, -24.1218414, 22.8650970, -46.9869385, 46.9869385
28: -18.7543030, 25.4579201, -18.7543030, 25.4579201, -44.0818634, 44.0821037
29: -25.6286564, 24.3507462, -25.6286564, 24.3507462, -49.9794006, 49.9794006
30: -25.7671089, 25.4406376, -25.7671089, 25.4406376, -51.2077484, 51.2077484
31: -21.9189606, 23.8200073, -21.9189606, 23.8200073, -45.7389679, 45.7389679
32: -36.1053810, 14.9813128, -36.1053810, 14.9813128, -49.6378822, 49.6369820
33: -50.8679428, 21.5127850, -50.8679428, 21.5127850, -69.9101334, 69.9079056
34: -50.5968628, 10.0608883, -50.5968628, 10.0608883, -56.7872467, 56.7839546
35: -42.9593925, 16.9696846, -42.9593925, 16.9696846, -57.5649338, 57.5632095
36: -41.9535332, 18.6489067, -41.9535332, 18.6489067, -60.5201263, 60.5199890
37: -55.7414169, 13.1964712, -55.7414169, 13.1964712, -67.8628540, 67.8614883
38: -52.8991852, 15.9395485, -52.8991852, 15.9395485, -68.8387299, 68.8387299
39: -61.7800980, 18.2122650, -61.7800980, 18.2122650, -79.5834045, 79.5828857
40: -48.4437256, 11.8220787, -48.4437256, 11.8220787, -59.2855225, 59.2845306
41: -35.9136467, 18.6692276, -35.9136467, 18.6692276, -53.4085922, 53.4080429
42: -26.5274906, 13.2349625, -26.5274906, 13.2349625, -38.2306061, 38.2298088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=362, inp2_unstable=362, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=25, inp2_unstable=25, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 560

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5329560, upper bound: 27.5355227
time: 61.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 31, lower bound: -27.5300624, upper bound: 27.5396442
time: 170.46 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 234.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 234.59
Output dim: 31, lower bound: -27.5413301, upper bound: 27.5305294
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 234.59
Output dim: 31, lower bound: -27.5452282, upper bound: 27.5269178
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 234.59
Output dim: 31, lower bound: -27.5320488, upper bound: 27.5389103
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 234.59
Output dim: 31, lower bound: -27.5367801, upper bound: 27.5341836
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 234.59
Output dim: 31, lower bound: -27.5245350, upper bound: 27.5161947
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 234.59
Output dim: 31, lower bound: -27.5261307, upper bound: 27.5146068
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 234.59
Output dim: 31, lower bound: -27.5141754, upper bound: 27.5250751
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 234.59
Output dim: 31, lower bound: -27.5169085, upper bound: 27.5223451
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 234.59
Output dim: 31, lower bound: -27.5312431, upper bound: 27.5299650
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 234.59
Output dim: 31, lower bound: -27.5297939, upper bound: 27.5314221
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 234.59
Output dim: 31, lower bound: -27.5149610, upper bound: 27.5313155
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 234.59
Output dim: 31, lower bound: -27.5309546, upper bound: 27.5153316
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 234.59
Output dim: 31, lower bound: -27.5329560, upper bound: 27.5355227
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 234.59
Output dim: 31, lower bound: -27.5300624, upper bound: 27.5396442
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 234.59
Output dim: 31, lower bound: -27.5344481, upper bound: 27.5423571

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 84.29 + 1922.62 = 2006.91 seconds
