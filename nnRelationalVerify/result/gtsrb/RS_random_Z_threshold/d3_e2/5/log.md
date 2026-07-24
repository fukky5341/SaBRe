## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 5)
Time budget: 7200 seconds
Split limit: 100
Threshold: 38.9746791072


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328)
1: (-31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664)
2: (-30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690)
3: (-34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942)
4: (-40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316)
5: (-37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755)
6: (-56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726)
7: (-43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831)
8: (-39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809)
9: (-34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275)
10: (-55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739)
11: (-56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778)
12: (-59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474)
13: (-48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827)
14: (-81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586)
15: (-40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336)
16: (-58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498)
17: (-85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696)
18: (-49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788)
19: (-41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662)
20: (-35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226)
21: (-49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371)
22: (-51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297)
23: (-39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429)
24: (-45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464)
25: (-38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314)
26: (-59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385)
27: (-49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097)
28: (-37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770)
29: (-55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838)
30: (-47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052)
31: (-49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653)
32: (-49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205)
33: (-72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273)
34: (-61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420)
35: (-57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084)
36: (-57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705)
37: (-85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175)
38: (-69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412)
39: (-85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074)
40: (-75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985)
41: (-54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703)
42: (-39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 3.12 + 109.86 = 112.98 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -39.0136928, upper bound: 39.0136928

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 976

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 590

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0129369, upper bound: 39.0092474
time: 68.50 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0092474, upper bound: 39.0129369
time: 84.11 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 152.63 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 152.63
Output dim: 2, lower bound: -39.0129369, upper bound: 39.0092474
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 152.63
Output dim: 2, lower bound: -39.0092474, upper bound: 39.0129369

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 638

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 666

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0099641, upper bound: 39.0075301
time: 75.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0112168, upper bound: 39.0062748
time: 80.16 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 586

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 708

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0073393, upper bound: 38.9972714
time: 79.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9935917, upper bound: 39.0110315
time: 70.49 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 152.27 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 152.27
Output dim: 2, lower bound: -39.0099641, upper bound: 39.0075301
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 152.27
Output dim: 2, lower bound: -39.0112168, upper bound: 39.0062748
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 152.27
Output dim: 2, lower bound: -39.0073393, upper bound: 38.9972714
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 152.27
Output dim: 2, lower bound: -38.9935917, upper bound: 39.0110315

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 713

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1563

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0045454, upper bound: 39.0020688
time: 84.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0045454, upper bound: 39.0020688
time: 93.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1543

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 633

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0104191, upper bound: 38.9526704
time: 224.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9570065, upper bound: 39.0054805
time: 71.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 649

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 728

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9427138, upper bound: 38.9962627
time: 83.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0063282, upper bound: 38.9326271
time: 78.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 597

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1563

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9881841, upper bound: 39.0056164
time: 78.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9881841, upper bound: 39.0056164
time: 79.71 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 160.43 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 160.43
Output dim: 2, lower bound: -39.0045454, upper bound: 39.0020688
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 160.43
Output dim: 2, lower bound: -39.0045454, upper bound: 39.0020688
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 160.43
Output dim: 2, lower bound: -39.0104191, upper bound: 38.9526704
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 160.43
Output dim: 2, lower bound: -38.9570065, upper bound: 39.0054805
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 160.43
Output dim: 2, lower bound: -38.9427138, upper bound: 38.9962627
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 160.43
Output dim: 2, lower bound: -39.0063282, upper bound: 38.9326271
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 160.43
Output dim: 2, lower bound: -38.9881841, upper bound: 39.0056164
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 160.43
Output dim: 2, lower bound: -38.9881841, upper bound: 39.0056164

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 550

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1540

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0035671, upper bound: 39.0010877
time: 65.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0035671, upper bound: 39.0019503
time: 70.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 751

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 658

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9864091, upper bound: 38.9995664
time: 72.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9983479, upper bound: 38.9839187
time: 81.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 711

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 653

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9727144, upper bound: 38.9521505
time: 75.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0098962, upper bound: 38.9150273
time: 79.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 719

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 709

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9545335, upper bound: 39.0049987
time: 127.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9564837, upper bound: 39.0029872
time: 73.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 686

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 591

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9425375, upper bound: 38.9907346
time: 72.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9375688, upper bound: 38.9961613
time: 79.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 701

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 674

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9870873, upper bound: 38.9272180
time: 85.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0008078, upper bound: 38.9272180
time: 75.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 963

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 601

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9768663, upper bound: 39.0015589
time: 81.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9841431, upper bound: 38.9942386
time: 159.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 742

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9738043, upper bound: 39.0048501
time: 88.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9874175, upper bound: 38.9912648
time: 80.17 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 171.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 171.24
Output dim: 2, lower bound: -39.0035671, upper bound: 39.0010877
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 171.24
Output dim: 2, lower bound: -39.0035671, upper bound: 39.0019503
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 171.24
Output dim: 2, lower bound: -38.9864091, upper bound: 38.9995664
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 171.24
Output dim: 2, lower bound: -38.9983479, upper bound: 38.9839187
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 171.24
Output dim: 2, lower bound: -38.9727144, upper bound: 38.9521505
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 171.24
Output dim: 2, lower bound: -39.0098962, upper bound: 38.9150273
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 171.24
Output dim: 2, lower bound: -38.9545335, upper bound: 39.0049987
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 171.24
Output dim: 2, lower bound: -38.9564837, upper bound: 39.0029872
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 171.24
Output dim: 2, lower bound: -38.9425375, upper bound: 38.9907346
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 171.24
Output dim: 2, lower bound: -38.9375688, upper bound: 38.9961613
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 171.24
Output dim: 2, lower bound: -38.9870873, upper bound: 38.9272180
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 171.24
Output dim: 2, lower bound: -39.0008078, upper bound: 38.9272180
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 171.24
Output dim: 2, lower bound: -38.9768663, upper bound: 39.0015589
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 171.24
Output dim: 2, lower bound: -38.9841431, upper bound: 38.9942386
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 171.24
Output dim: 2, lower bound: -38.9738043, upper bound: 39.0048501
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 171.24
Output dim: 2, lower bound: -38.9874175, upper bound: 38.9912648

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1281

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 646

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0037474, upper bound: 38.9563302
time: 124.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9587917, upper bound: 39.0004111
time: 80.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 691

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 565

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9883565, upper bound: 38.9998165
time: 100.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9977076, upper bound: 38.9867361
time: 122.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 634

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9853894, upper bound: 38.9987161
time: 80.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9817793, upper bound: 38.9986310
time: 100.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 710

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 613

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9850991, upper bound: 38.9835773
time: 81.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0017552, upper bound: 38.9706272
time: 120.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 773

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 779

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0096763, upper bound: 38.9133931
time: 89.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0082973, upper bound: 38.9148039
time: 98.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 653

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 683

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9136798, upper bound: 39.0039610
time: 71.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9536018, upper bound: 38.9650868
time: 77.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 711

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1330

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9560652, upper bound: 39.0025723
time: 76.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9560687, upper bound: 39.0025672
time: 93.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 598

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 725

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9240571, upper bound: 38.9872504
time: 127.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9392324, upper bound: 38.9860127
time: 87.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1491

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 522

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9345943, upper bound: 38.9916566
time: 92.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9329990, upper bound: 38.9932077
time: 79.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 613

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 615

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0002040, upper bound: 38.9243249
time: 85.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9979565, upper bound: 38.9265998
time: 90.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 777

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9708421, upper bound: 38.9260705
time: 71.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9996517, upper bound: 38.8974058
time: 98.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 741

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 695

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9116614, upper bound: 38.9995635
time: 77.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9748565, upper bound: 38.9363151
time: 81.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 779

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 773

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9840100, upper bound: 38.9935248
time: 78.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9834188, upper bound: 38.9941092
time: 109.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 626

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 652

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9503554, upper bound: 39.0043558
time: 75.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9733111, upper bound: 38.9813791
time: 85.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1491

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1649

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9863930, upper bound: 38.9790276
time: 80.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9751940, upper bound: 38.9902373
time: 81.89 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 164.83 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -39.0037474, upper bound: 38.9563302
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -38.9587917, upper bound: 39.0004111
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -38.9883565, upper bound: 38.9998165
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -38.9977076, upper bound: 38.9867361
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -38.9853894, upper bound: 38.9987161
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -38.9817793, upper bound: 38.9986310
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -38.9850991, upper bound: 38.9835773
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -39.0017552, upper bound: 38.9706272
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -39.0096763, upper bound: 38.9133931
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -39.0082973, upper bound: 38.9148039
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -38.9136798, upper bound: 39.0039610
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 164.83
Output dim: 2, lower bound: -38.9536018, upper bound: 38.9650868
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -38.9560652, upper bound: 39.0025723
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -38.9560687, upper bound: 39.0025672
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -38.9240571, upper bound: 38.9872504
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -38.9392324, upper bound: 38.9860127
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -38.9345943, upper bound: 38.9916566
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -38.9329990, upper bound: 38.9932077
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -39.0002040, upper bound: 38.9243249
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -38.9979565, upper bound: 38.9265998
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 164.83
Output dim: 2, lower bound: -38.9708421, upper bound: 38.9260705
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -38.9996517, upper bound: 38.8974058
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -38.9116614, upper bound: 38.9995635
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -38.9748565, upper bound: 38.9363151
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -38.9840100, upper bound: 38.9935248
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -38.9834188, upper bound: 38.9941092
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -38.9503554, upper bound: 39.0043558
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -38.9733111, upper bound: 38.9813791
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -38.9863930, upper bound: 38.9790276
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 164.83
Output dim: 2, lower bound: -38.9751940, upper bound: 38.9902373

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 661

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 713

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9542433, upper bound: 38.9548727
time: 85.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0023303, upper bound: 38.9075165
time: 96.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 595

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1771

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9584773, upper bound: 38.9993012
time: 69.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9585033, upper bound: 39.0000573
time: 75.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 536

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 557

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9826219, upper bound: 38.9998106
time: 105.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9846205, upper bound: 38.9943381
time: 74.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 718

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 634

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0004049, upper bound: 38.9858565
time: 67.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9968051, upper bound: 38.9857591
time: 73.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 571

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 679

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9734458, upper bound: 38.9959201
time: 70.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9825568, upper bound: 38.9867976
time: 78.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 732

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 751

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9848936, upper bound: 38.9828370
time: 90.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9697172, upper bound: 38.9980130
time: 75.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 521

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 714

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9516789, upper bound: 38.9829758
time: 70.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9881860, upper bound: 38.9463687
time: 82.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 683

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1763

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9689087, upper bound: 38.9377821
time: 72.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9689732, upper bound: 38.9377171
time: 77.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 677

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1547

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9995018, upper bound: 38.9045256
time: 78.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9995018, upper bound: 38.9045256
time: 86.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 976

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 605

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9713208, upper bound: 38.9137555
time: 84.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0035393, upper bound: 38.8772723
time: 91.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -53.5198631, 43.0871696, -53.5198631, 43.0871696, -96.6070328, 96.6070328
1: -31.7694683, 36.1152000, -31.7694683, 36.1152000, -67.8846664, 67.8846664
2: -30.5395241, 35.6862411, -30.5395241, 35.6862411, -66.2257690, 66.2257690
3: -34.0840378, 41.6862526, -34.0840378, 41.6862526, -75.7702866, 75.7702942
4: -40.1819534, 39.0007782, -40.1819534, 39.0007782, -79.1827240, 79.1827316
5: -37.0289001, 41.4712830, -37.0289001, 41.4712830, -78.5001831, 78.5001755
6: -56.0298233, 22.5602531, -56.0298233, 22.5602531, -78.5900726, 78.5900726
7: -43.0932312, 40.2791595, -43.0932312, 40.2791595, -83.3723907, 83.3723831
8: -39.5297775, 45.6106033, -39.5297775, 45.6106033, -85.1403809, 85.1403809
9: -34.3088684, 37.5730629, -34.3088684, 37.5730629, -71.8819275, 71.8819275
10: -55.3341293, 52.4759445, -55.3341293, 52.4759445, -107.8100739, 107.8100739
11: -56.6364098, 39.8233643, -56.6364098, 39.8233643, -96.4597778, 96.4597778
12: -59.2651138, 44.1722374, -59.2651138, 44.1722374, -103.4373474, 103.4373474
13: -48.8810768, 49.7143173, -48.8810768, 49.7143173, -98.5953827, 98.5953827
14: -81.7089539, 43.4845047, -81.7089539, 43.4845047, -125.1934586, 125.1934586
15: -40.5275764, 36.4627571, -40.5275764, 36.4627571, -76.9903336, 76.9903336
16: -58.4362526, 40.9439049, -58.4362526, 40.9439049, -99.3801498, 99.3801498
17: -85.3832779, 62.6455994, -85.3832779, 62.6455994, -148.0288696, 148.0288696
18: -49.1222458, 29.2441349, -49.1222458, 29.2441349, -78.3663788, 78.3663788
19: -41.4818268, 19.5866394, -41.4818268, 19.5866394, -61.0684662, 61.0684662
20: -35.4976654, 21.8792553, -35.4976654, 21.8792553, -57.3769226, 57.3769226
21: -49.3102570, 25.5471878, -49.3102570, 25.5471878, -74.8574371, 74.8574371
22: -51.1287766, 30.1950531, -51.1287766, 30.1950531, -81.3238297, 81.3238297
23: -39.2706413, 26.6971016, -39.2706413, 26.6971016, -65.9677429, 65.9677429
24: -45.3573952, 22.9401550, -45.3573952, 22.9401550, -68.2975464, 68.2975464
25: -38.6473312, 31.1510963, -38.6473312, 31.1510963, -69.7984314, 69.7984314
26: -59.2387047, 37.7995377, -59.2387047, 37.7995377, -97.0382385, 97.0382385
27: -49.5083771, 27.4447308, -49.5083771, 27.4447308, -76.9530945, 76.9531097
28: -37.9644432, 28.9301300, -37.9644432, 28.9301300, -66.8945770, 66.8945770
29: -55.6128922, 34.4748955, -55.6128922, 34.4748955, -90.0877838, 90.0877838
30: -47.9148865, 27.3326206, -47.9148865, 27.3326206, -75.2475052, 75.2475052
31: -49.1540298, 24.1216393, -49.1540298, 24.1216393, -73.2756653, 73.2756653
32: -49.2517548, 27.5498695, -49.2517548, 27.5498695, -76.8016205, 76.8016205
33: -72.0044022, 44.1667252, -72.0044022, 44.1667252, -116.1711273, 116.1711273
34: -61.0236931, 30.1596489, -61.0236931, 30.1596489, -91.1833420, 91.1833420
35: -57.3259888, 34.8101196, -57.3259888, 34.8101196, -92.1361084, 92.1361084
36: -57.3891716, 34.0602989, -57.3891716, 34.0602989, -91.4494705, 91.4494705
37: -85.4126892, 33.2540283, -85.4126892, 33.2540283, -118.6667023, 118.6667175
38: -69.2199783, 41.0800705, -69.2199783, 41.0800705, -110.3000488, 110.3000412
39: -85.1752548, 40.8695564, -85.1752548, 40.8695564, -126.0448151, 126.0448074
40: -75.3201828, 30.0895157, -75.3201828, 30.0895157, -105.4096985, 105.4096985
41: -54.4162827, 26.0835953, -54.4162827, 26.0835953, -80.4998779, 80.4998703
42: -39.0124054, 29.5149136, -39.0124054, 29.5149136, -68.5273209, 68.5273209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1542

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 680

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9119911, upper bound: 39.0032364
time: 72.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9129619, upper bound: 39.0022624
time: 76.74 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 151.80 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 151.80
Output dim: 2, lower bound: -38.9542433, upper bound: 38.9548727
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 151.80
Output dim: 2, lower bound: -39.0023303, upper bound: 38.9075165
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 151.80
Output dim: 2, lower bound: -38.9584773, upper bound: 38.9993012
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 151.80
Output dim: 2, lower bound: -38.9585033, upper bound: 39.0000573
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 151.80
Output dim: 2, lower bound: -38.9826219, upper bound: 38.9998106
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 151.80
Output dim: 2, lower bound: -38.9846205, upper bound: 38.9943381
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 151.80
Output dim: 2, lower bound: -39.0004049, upper bound: 38.9858565
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 151.80
Output dim: 2, lower bound: -38.9968051, upper bound: 38.9857591
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 151.80
Output dim: 2, lower bound: -38.9734458, upper bound: 38.9959201
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 151.80
Output dim: 2, lower bound: -38.9825568, upper bound: 38.9867976
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 151.80
Output dim: 2, lower bound: -38.9848936, upper bound: 38.9828370
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 151.80
Output dim: 2, lower bound: -38.9697172, upper bound: 38.9980130
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 151.80
Output dim: 2, lower bound: -38.9516789, upper bound: 38.9829758
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 151.80
Output dim: 2, lower bound: -38.9881860, upper bound: 38.9463687
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 151.80
Output dim: 2, lower bound: -38.9689087, upper bound: 38.9377821
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 151.80
Output dim: 2, lower bound: -38.9689732, upper bound: 38.9377171
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 151.80
Output dim: 2, lower bound: -38.9995018, upper bound: 38.9045256
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 151.80
Output dim: 2, lower bound: -38.9995018, upper bound: 38.9045256
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 151.80
Output dim: 2, lower bound: -38.9713208, upper bound: 38.9137555
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 151.80
Output dim: 2, lower bound: -39.0035393, upper bound: 38.8772723
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 151.80
Output dim: 2, lower bound: -38.9119911, upper bound: 39.0032364
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 151.80
Output dim: 2, lower bound: -38.9129619, upper bound: 39.0022624
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 151.80
Output dim: 2, lower bound: -38.9560652, upper bound: 39.0025723
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 151.80
Output dim: 2, lower bound: -38.9560687, upper bound: 39.0025672
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 151.80
Output dim: 2, lower bound: -38.9240571, upper bound: 38.9872504
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 151.80
Output dim: 2, lower bound: -38.9392324, upper bound: 38.9860127
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 151.80
Output dim: 2, lower bound: -38.9345943, upper bound: 38.9916566
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 151.80
Output dim: 2, lower bound: -38.9329990, upper bound: 38.9932077
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 151.80
Output dim: 2, lower bound: -39.0002040, upper bound: 38.9243249
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 151.80
Output dim: 2, lower bound: -38.9979565, upper bound: 38.9265998
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 151.80
Output dim: 2, lower bound: -38.9996517, upper bound: 38.8974058
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 151.80
Output dim: 2, lower bound: -38.9116614, upper bound: 38.9995635
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 151.80
Output dim: 2, lower bound: -38.9748565, upper bound: 38.9363151
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 151.80
Output dim: 2, lower bound: -38.9840100, upper bound: 38.9935248
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 151.80
Output dim: 2, lower bound: -38.9834188, upper bound: 38.9941092
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 151.80
Output dim: 2, lower bound: -38.9503554, upper bound: 39.0043558
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 151.80
Output dim: 2, lower bound: -38.9733111, upper bound: 38.9813791
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 151.80
Output dim: 2, lower bound: -38.9863930, upper bound: 38.9790276
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 151.80
Output dim: 2, lower bound: -38.9751940, upper bound: 38.9902373

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 112.98 + 7186.18 = 7299.15 seconds
