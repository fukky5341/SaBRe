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
execution time: IAR + RelationalAnalysis = 2.81 + 103.80 = 106.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -39.0136928, upper bound: 39.0136928

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 637
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 729
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 1491
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 645
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 637

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9393995, upper bound: 39.0022293
time: 82.60 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -39.0126807, upper bound: 39.0126809
time: 77.46 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 160.21 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 160.21
Output dim: 2, lower bound: -38.9393995, upper bound: 39.0022293
IS_A2, status: Status.UNKNOWN, split count: 1, time: 160.21
Output dim: 2, lower bound: -39.0126807, upper bound: 39.0126809

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -53.3692322, 43.0158386, -53.4772491, 43.0799179, -96.4491425, 96.4930878
1: -31.6396675, 36.0396385, -31.7270927, 36.1082916, -67.7479553, 67.7667313
2: -30.4028244, 35.6051941, -30.4946270, 35.6798172, -66.0826416, 66.0998230
3: -33.9130402, 41.5707703, -34.0269852, 41.6771049, -75.5901489, 75.5977554
4: -40.0292435, 38.8992386, -40.1312103, 38.9915848, -79.0208282, 79.0304413
5: -36.8595123, 41.3587494, -36.9719009, 41.4610901, -78.3206024, 78.3306427
6: -55.9504700, 22.5105400, -56.0119514, 22.5458755, -78.4963455, 78.5224915
7: -42.9266586, 40.1937141, -43.0368538, 40.2714920, -83.1981430, 83.2305679
8: -39.3845444, 45.5217285, -39.4825134, 45.6020660, -84.9866028, 85.0042419
9: -34.1658974, 37.5115280, -34.2646713, 37.5634613, -71.7293549, 71.7761993
10: -55.2284660, 52.3492393, -55.3063507, 52.4374123, -107.6658783, 107.6555862
11: -56.5202980, 39.7039566, -56.6194839, 39.7855606, -96.3058624, 96.3234406
12: -59.1601830, 44.0383759, -59.2532616, 44.1295166, -103.2896881, 103.2916336
13: -48.7329178, 49.6134186, -48.8336182, 49.6984406, -98.4313583, 98.4470367
14: -81.5915833, 43.3708534, -81.6826019, 43.4491463, -125.0407257, 125.0534515
15: -40.4264336, 36.4005318, -40.4951859, 36.4524307, -76.8788605, 76.8957138
16: -58.2762871, 40.9023819, -58.3915825, 40.9341850, -99.2104721, 99.2939606
17: -85.2607574, 62.5175858, -85.3532867, 62.6078033, -147.8685608, 147.8708801
18: -49.0410042, 29.0890388, -49.1070404, 29.1945915, -78.2355957, 78.1960754
19: -41.3808327, 19.4679756, -41.4705086, 19.5465889, -60.9274216, 60.9384842
20: -35.4361267, 21.7760887, -35.4861450, 21.8473759, -57.2835007, 57.2622337
21: -49.2111435, 25.4312534, -49.2955780, 25.5089264, -74.7200699, 74.7268295
22: -51.0256386, 30.0682335, -51.1137657, 30.1541176, -81.1797562, 81.1819992
23: -39.1495514, 26.5407352, -39.2590714, 26.6443787, -65.7939301, 65.7998047
24: -45.2441483, 22.8083763, -45.3451424, 22.8973866, -68.1415329, 68.1535187
25: -38.5518494, 31.0021286, -38.6352844, 31.1026688, -69.6545181, 69.6374054
26: -59.1315346, 37.6063232, -59.2252464, 37.7372513, -96.8687820, 96.8315735
27: -49.4049683, 27.3332996, -49.4943886, 27.4077187, -76.8126831, 76.8276825
28: -37.8699799, 28.7891464, -37.9548569, 28.8840904, -66.7540741, 66.7440033
29: -55.4897194, 34.3542328, -55.5956345, 34.4356651, -89.9253769, 89.9498672
30: -47.8279877, 27.2370720, -47.9025497, 27.3029652, -75.1309509, 75.1396179
31: -49.0196648, 23.9912567, -49.1385994, 24.0785599, -73.0982132, 73.1298523
32: -49.1631660, 27.4759102, -49.2372131, 27.5260315, -76.6891937, 76.7131195
33: -71.9232635, 44.0810242, -71.9854736, 44.1409798, -116.0642395, 116.0664978
34: -60.9337769, 30.0178795, -61.0095062, 30.1126766, -91.0464478, 91.0273895
35: -57.2636337, 34.7182846, -57.3128281, 34.7807541, -92.0443878, 92.0311127
36: -57.3285980, 33.9530487, -57.3780861, 34.0242615, -91.3528595, 91.3311310
37: -85.2532425, 33.0981674, -85.3896332, 33.2030487, -118.4562759, 118.4878006
38: -69.1333313, 40.9355812, -69.2031708, 41.0345306, -110.1678619, 110.1387482
39: -85.0612640, 40.7875366, -85.1515808, 40.8427505, -125.9040070, 125.9391174
40: -75.2287216, 30.0363064, -75.3014450, 30.0730495, -105.3017731, 105.3377533
41: -54.3191528, 25.9901714, -54.4019775, 26.0527534, -80.3719025, 80.3921509
42: -38.9437485, 29.4344749, -38.9994164, 29.4891739, -68.4329224, 68.4338913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 599
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1491
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 645
type: B, layer: 1, pos: 645
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 603

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.8792107, upper bound: 38.9962280
time: 76.68 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.8792107, upper bound: 39.0002868
time: 88.22 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -53.5110130, 43.0840836, -53.5167999, 43.0861549, -96.5971680, 96.6008835
1: -31.7621670, 36.1126175, -31.7670097, 36.1143456, -67.8765106, 67.8796234
2: -30.5331554, 35.6838608, -30.5373993, 35.6854553, -66.2186050, 66.2212601
3: -34.0760231, 41.6832275, -34.0813713, 41.6852646, -75.7612915, 75.7645950
4: -40.1744576, 38.9971466, -40.1794548, 38.9995308, -79.1739883, 79.1765976
5: -37.0204544, 41.4684677, -37.0260620, 41.4703331, -78.4907837, 78.4945221
6: -56.0138588, 22.5566921, -56.0247383, 22.5590057, -78.5728607, 78.5814285
7: -43.0838737, 40.2764740, -43.0900726, 40.2782631, -83.3621292, 83.3665466
8: -39.5217018, 45.6064949, -39.5270805, 45.6092606, -85.1309662, 85.1335754
9: -34.3015976, 37.5690231, -34.3063889, 37.5717163, -71.8733139, 71.8754120
10: -55.3277664, 52.4685860, -55.3319206, 52.4735794, -107.8013458, 107.8004913
11: -56.6304703, 39.8177185, -56.6343765, 39.8215485, -96.4520187, 96.4520950
12: -59.2606964, 44.1656303, -59.2636642, 44.1699944, -103.4306870, 103.4292908
13: -48.8736191, 49.7088013, -48.8786469, 49.7124825, -98.5861053, 98.5874481
14: -81.7012329, 43.4790497, -81.7063065, 43.4827042, -125.1839371, 125.1853561
15: -40.5210800, 36.4594994, -40.5253792, 36.4617081, -76.9827881, 76.9848785
16: -58.4274864, 40.9406128, -58.4332695, 40.9428329, -99.3703156, 99.3738861
17: -85.3746338, 62.6387634, -85.3804169, 62.6433868, -148.0180054, 148.0191650
18: -49.1169853, 29.2361813, -49.1205177, 29.2415257, -78.3585052, 78.3566971
19: -41.4777641, 19.5813961, -41.4804993, 19.5849342, -61.0626984, 61.0618935
20: -35.4940567, 21.8737774, -35.4964409, 21.8774014, -57.3714600, 57.3702164
21: -49.3043938, 25.5421314, -49.3083153, 25.5455303, -74.8499222, 74.8504486
22: -51.1230087, 30.1893806, -51.1268845, 30.1931648, -81.3161697, 81.3162689
23: -39.2666817, 26.6898842, -39.2692490, 26.6947346, -65.9614105, 65.9591293
24: -45.3527489, 22.9337273, -45.3558273, 22.9380360, -68.2907791, 68.2895508
25: -38.6432190, 31.1433754, -38.6459503, 31.1484833, -69.7916870, 69.7893219
26: -59.2334366, 37.7908096, -59.2369843, 37.7966499, -97.0300903, 97.0277939
27: -49.5019684, 27.4397697, -49.5062637, 27.4431305, -76.9450989, 76.9460297
28: -37.9612389, 28.9232941, -37.9633636, 28.9278183, -66.8890533, 66.8866577
29: -55.6053085, 34.4697952, -55.6104126, 34.4732399, -90.0785370, 90.0802078
30: -47.9094467, 27.3279667, -47.9130630, 27.3310680, -75.2405090, 75.2410278
31: -49.1488647, 24.1154308, -49.1523132, 24.1195946, -73.2684631, 73.2677460
32: -49.2473030, 27.5437126, -49.2502708, 27.5478230, -76.7951279, 76.7939758
33: -71.9955597, 44.1615219, -72.0015259, 44.1649818, -116.1605377, 116.1630478
34: -61.0199509, 30.1520576, -61.0224152, 30.1571121, -91.1770630, 91.1744690
35: -57.3226814, 34.8046379, -57.3248901, 34.8082504, -92.1309357, 92.1295242
36: -57.3854942, 34.0540428, -57.3879852, 34.0581741, -91.4436646, 91.4420319
37: -85.4057922, 33.2468948, -85.4103546, 33.2517014, -118.6574936, 118.6572495
38: -69.2155914, 41.0711670, -69.2185440, 41.0770912, -110.2926788, 110.2897110
39: -85.1699448, 40.8639145, -85.1734619, 40.8676682, -126.0376129, 126.0373611
40: -75.3141861, 30.0862923, -75.3181763, 30.0884590, -105.4026489, 105.4044647
41: -54.4108162, 26.0784569, -54.4144287, 26.0819035, -80.4927216, 80.4928894
42: -39.0076828, 29.5094452, -39.0107803, 29.5127335, -68.5204163, 68.5202255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 603
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 729
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 599
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 603

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9331894, upper bound: 38.9532812
time: 73.63 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.9331894, upper bound: 38.9532812
time: 155.03 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 231.10 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 231.10
Output dim: 2, lower bound: -38.8792107, upper bound: 38.9962280
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 231.10
Output dim: 2, lower bound: -38.8792107, upper bound: 39.0002868
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 231.10
Output dim: 2, lower bound: -38.9331894, upper bound: 38.9532812
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 231.10
Output dim: 2, lower bound: -38.9331894, upper bound: 38.9532812

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -53.1840134, 42.9238358, -53.4186401, 43.0678101, -96.2518158, 96.3424759
1: -31.5139103, 35.9596863, -31.6850796, 36.0984421, -67.6123505, 67.6447601
2: -30.2697258, 35.5180168, -30.4496822, 35.6695480, -65.9392700, 65.9676971
3: -33.7690353, 41.4692078, -33.9780006, 41.6643295, -75.4333649, 75.4472046
4: -39.8733978, 38.7964096, -40.0786743, 38.9802895, -78.8536835, 78.8750839
5: -36.7220764, 41.2507324, -36.9248962, 41.4473190, -78.1693954, 78.1756287
6: -55.8830566, 22.3770943, -55.9979248, 22.5015831, -78.3846436, 78.3750153
7: -42.7854691, 40.1153908, -42.9891701, 40.2610207, -83.0464935, 83.1045532
8: -39.2273254, 45.4084930, -39.4294815, 45.5878983, -84.8152161, 84.8379745
9: -34.0208664, 37.4672432, -34.2178764, 37.5514145, -71.5722809, 71.6851196
10: -55.1284752, 52.2122116, -55.2793961, 52.3931084, -107.5215836, 107.4916000
11: -56.4044075, 39.5412025, -56.5986671, 39.7306519, -96.1350555, 96.1398697
12: -59.0261497, 43.8412170, -59.2377129, 44.0641251, -103.0902710, 103.0789261
13: -48.5817299, 49.5299416, -48.7884521, 49.6738548, -98.2555847, 98.3183899
14: -81.4686508, 43.2845154, -81.6465912, 43.4201660, -124.8888168, 124.9311066
15: -40.2645035, 36.3386917, -40.4420929, 36.4389267, -76.7034225, 76.7807846
16: -58.1199760, 40.8421097, -58.3451920, 40.9174156, -99.0373840, 99.1872940
17: -85.1300659, 62.4448090, -85.3124924, 62.5866241, -147.7166901, 147.7572937
18: -48.9488602, 28.9054375, -49.0807419, 29.1342373, -78.0830994, 77.9861755
19: -41.3169250, 19.3724365, -41.4572144, 19.5143738, -60.8312988, 60.8296394
20: -35.3623886, 21.6509285, -35.4709015, 21.8062305, -57.1686172, 57.1218262
21: -49.1250305, 25.3261337, -49.2753181, 25.4740753, -74.5991058, 74.6014481
22: -50.9448967, 29.9469547, -51.0905685, 30.1144123, -81.0593109, 81.0375214
23: -39.0541763, 26.4133759, -39.2453308, 26.6016769, -65.6558533, 65.6587067
24: -45.1664467, 22.7166500, -45.3275108, 22.8670197, -68.0334625, 68.0441589
25: -38.4779816, 30.8880978, -38.6161880, 31.0654469, -69.5434189, 69.5042801
26: -59.0083122, 37.3778610, -59.2033806, 37.6621628, -96.6704712, 96.5812378
27: -49.3109398, 27.1860714, -49.4721909, 27.3588619, -76.6697998, 76.6582642
28: -37.8134460, 28.6725483, -37.9400978, 28.8456879, -66.6591339, 66.6126404
29: -55.4103012, 34.2423172, -55.5723534, 34.3970871, -89.8073883, 89.8146667
30: -47.7375336, 27.1360645, -47.8763351, 27.2705193, -75.0080414, 75.0123978
31: -48.9198723, 23.8754158, -49.1192741, 24.0404510, -72.9603271, 72.9946899
32: -49.0368958, 27.2867298, -49.2193871, 27.4623985, -76.4992981, 76.5061188
33: -71.8197098, 44.0197906, -71.9585419, 44.1219444, -115.9416351, 115.9783325
34: -60.8349609, 29.8571739, -60.9908562, 30.0589581, -90.8939209, 90.8480225
35: -57.1881180, 34.6197510, -57.2940178, 34.7491722, -91.9372864, 91.9137726
36: -57.2046242, 33.7496719, -57.3610535, 33.9544296, -91.1590576, 91.1107178
37: -85.1200943, 32.9800301, -85.3633728, 33.1627693, -118.2828598, 118.3433990
38: -68.9681854, 40.6575699, -69.1784515, 40.9410629, -109.9092484, 109.8360214
39: -84.9319916, 40.6718864, -85.1243286, 40.8026848, -125.7346802, 125.7962189
40: -75.1404114, 29.9546700, -75.2778397, 30.0461311, -105.1865387, 105.2324982
41: -54.2098160, 25.8244667, -54.3861618, 25.9972839, -80.2070999, 80.2106247
42: -38.8764229, 29.2954578, -38.9884949, 29.4433060, -68.3197250, 68.2839508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=405, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 729
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 599
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1491
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 729

## Relational analysis of IS_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.8071526, upper bound: 38.9940889
time: 73.70 seconds

## Relational analysis of IS_A1_A1_A2

### Relational analysis result of IS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.8071526, upper bound: 38.9950159
time: 79.25 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -53.3575630, 43.0117149, -53.4734230, 43.0785522, -96.4361115, 96.4851379
1: -31.6326962, 36.0358734, -31.7248192, 36.1070366, -67.7397308, 67.7606964
2: -30.3963566, 35.6011467, -30.4925251, 35.6784744, -66.0748291, 66.0936737
3: -33.9051132, 41.5670738, -34.0244179, 41.6759109, -75.5810242, 75.5914917
4: -40.0222473, 38.8934021, -40.1289444, 38.9896774, -79.0119247, 79.0223389
5: -36.8526115, 41.3551559, -36.9696579, 41.4599152, -78.3125305, 78.3248138
6: -55.9426613, 22.5042915, -56.0093498, 22.5438652, -78.4865265, 78.5136414
7: -42.9190216, 40.1895905, -43.0343399, 40.2701149, -83.1891327, 83.2239304
8: -39.3769989, 45.5154114, -39.4800415, 45.5999870, -84.9769897, 84.9954529
9: -34.1573219, 37.5066452, -34.2618561, 37.5618896, -71.7192001, 71.7685013
10: -55.2182770, 52.3392181, -55.3030434, 52.4341240, -107.6524048, 107.6422501
11: -56.5138664, 39.6959457, -56.6173820, 39.7829742, -96.2968445, 96.3133240
12: -59.1532211, 44.0293846, -59.2509766, 44.1266403, -103.2798615, 103.2803650
13: -48.7241898, 49.5983696, -48.8307724, 49.6935844, -98.4177704, 98.4291382
14: -81.5819092, 43.3519592, -81.6794052, 43.4423561, -125.0242615, 125.0313644
15: -40.4176369, 36.3969040, -40.4922638, 36.4512596, -76.8688889, 76.8891602
16: -58.2665215, 40.8820114, -58.3883438, 40.9272079, -99.1937256, 99.2703552
17: -85.2493210, 62.4616852, -85.3495255, 62.5902672, -147.8395691, 147.8112183
18: -49.0336418, 29.0802269, -49.1046219, 29.1912880, -78.2249222, 78.1848450
19: -41.3760910, 19.4629440, -41.4689331, 19.5449562, -60.9210472, 60.9318771
20: -35.4319763, 21.7684288, -35.4847832, 21.8448639, -57.2768364, 57.2532120
21: -49.2049713, 25.4254131, -49.2935410, 25.5070305, -74.7120056, 74.7189484
22: -51.0180054, 30.0611153, -51.1112289, 30.1517715, -81.1697769, 81.1723404
23: -39.1449966, 26.5348244, -39.2575684, 26.6424599, -65.7874603, 65.7923889
24: -45.2392464, 22.8018932, -45.3435211, 22.8952332, -68.1344757, 68.1454163
25: -38.5476875, 30.9951534, -38.6339340, 31.1003551, -69.6480408, 69.6290894
26: -59.1249008, 37.5955544, -59.2230644, 37.7338142, -96.8587112, 96.8186111
27: -49.3962479, 27.3271523, -49.4914780, 27.4056206, -76.8018646, 76.8186188
28: -37.8650589, 28.7842293, -37.9532089, 28.8824883, -66.7475433, 66.7374420
29: -55.4803123, 34.3449097, -55.5925102, 34.4321518, -89.9124603, 89.9374237
30: -47.8185272, 27.2305584, -47.8994370, 27.3008118, -75.1193314, 75.1299896
31: -49.0139923, 23.9852467, -49.1367264, 24.0765953, -73.0905914, 73.1219711
32: -49.1549644, 27.4669762, -49.2344856, 27.5231476, -76.6781082, 76.7014618
33: -71.8637619, 44.0746155, -71.9667511, 44.1388321, -116.0025864, 116.0413666
34: -60.9263458, 30.0098877, -61.0070190, 30.1100616, -91.0364075, 91.0169067
35: -57.2474594, 34.7129517, -57.3076057, 34.7790337, -92.0264816, 92.0205536
36: -57.3203697, 33.9437714, -57.3753471, 34.0212784, -91.3416367, 91.3191223
37: -85.2457962, 33.0919342, -85.3872070, 33.2010384, -118.4468384, 118.4791412
38: -69.1238098, 40.9220734, -69.2000122, 41.0301132, -110.1539001, 110.1220856
39: -85.0549011, 40.7798576, -85.1495361, 40.8402634, -125.8951569, 125.9293976
40: -75.2207947, 30.0315514, -75.2988434, 30.0715103, -105.2922974, 105.3303909
41: -54.3109665, 25.9829006, -54.3992538, 26.0504398, -80.3614044, 80.3821564
42: -38.9377518, 29.4277954, -38.9974213, 29.4870186, -68.4247589, 68.4252167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 729
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 599
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 729

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.8770874, upper bound: 38.9284062
time: 68.01 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.9362458, upper bound: 38.9990714
time: 75.60 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 146.02 seconds
IS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 146.02
Output dim: 2, lower bound: -38.8071526, upper bound: 38.9940889
IS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 146.02
Output dim: 2, lower bound: -38.8071526, upper bound: 38.9950159
IS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 146.02
Output dim: 2, lower bound: -38.8770874, upper bound: 38.9284062
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 146.02
Output dim: 2, lower bound: -38.9362458, upper bound: 38.9990714

## BFS IS instance: IS_A1_A1_A1

### Backsubstitution after applying IS history:
0: -52.9642906, 42.8381348, -53.3481483, 43.0573959, -96.0216827, 96.1862717
1: -31.3563805, 35.8879700, -31.6307125, 36.0890808, -67.4454651, 67.5186768
2: -30.1098213, 35.4355278, -30.3945961, 35.6605682, -65.7703857, 65.8301239
3: -33.6086731, 41.3835297, -33.9210281, 41.6528931, -75.2615662, 75.3045578
4: -39.7320976, 38.7168617, -40.0300560, 38.9707336, -78.7028351, 78.7469177
5: -36.5661087, 41.1616325, -36.8700333, 41.4359665, -78.0020752, 78.0316620
6: -55.8297806, 22.3062096, -55.9854813, 22.4713745, -78.3011475, 78.2916870
7: -42.5677910, 39.9964142, -42.9115982, 40.2453651, -82.8131561, 82.9080048
8: -38.9935074, 45.2687798, -39.3466263, 45.5730057, -84.5665131, 84.6154022
9: -33.9326706, 37.4127922, -34.1878090, 37.5391922, -71.4718628, 71.6006012
10: -55.0273628, 52.1530724, -55.2470856, 52.3762093, -107.4035721, 107.4001541
11: -56.2997932, 39.4541283, -56.5731468, 39.7013664, -96.0011597, 96.0272751
12: -58.9423141, 43.6745758, -59.2251511, 44.0081100, -102.9504242, 102.8997269
13: -48.5099907, 49.4495621, -48.7618103, 49.6524124, -98.1623993, 98.2113724
14: -81.2120285, 43.1765442, -81.5605316, 43.4105453, -124.6225739, 124.7370682
15: -40.1548882, 36.2793503, -40.3978348, 36.4225578, -76.5774460, 76.6771851
16: -58.0142746, 40.7862930, -58.3140488, 40.8949509, -98.9092255, 99.1003418
17: -84.9962006, 62.3574295, -85.2726517, 62.5572166, -147.5534210, 147.6300659
18: -48.8523712, 28.8203659, -49.0633736, 29.1045074, -77.9568787, 77.8837433
19: -41.2379150, 19.2851219, -41.4436646, 19.4835110, -60.7214241, 60.7287827
20: -35.3045654, 21.5977154, -35.4535904, 21.7863464, -57.0909119, 57.0512962
21: -49.0422745, 25.2502098, -49.2582207, 25.4487915, -74.4910660, 74.5084305
22: -50.8150482, 29.7817307, -51.0604286, 30.0553703, -80.8704147, 80.8421478
23: -38.9618645, 26.3263550, -39.2301712, 26.5720463, -65.5339050, 65.5565186
24: -45.0411034, 22.6224022, -45.3028259, 22.8337536, -67.8748550, 67.9252319
25: -38.3637009, 30.7350864, -38.5941162, 31.0113983, -69.3750992, 69.3291931
26: -58.8651047, 37.1586571, -59.1780434, 37.5840836, -96.4491882, 96.3367004
27: -49.2370186, 27.1483574, -49.4516068, 27.3416233, -76.5786362, 76.5999603
28: -37.7292557, 28.5810852, -37.9277229, 28.8150463, -66.5443039, 66.5088043
29: -55.2623558, 34.1071472, -55.5407410, 34.3492355, -89.6115875, 89.6478806
30: -47.6161652, 27.0466480, -47.8529243, 27.2417736, -74.8579254, 74.8995667
31: -48.8187752, 23.8060856, -49.0998268, 24.0169888, -72.8357544, 72.9059143
32: -48.9842491, 27.2270050, -49.2042961, 27.4342270, -76.4184647, 76.4313049
33: -71.7106705, 43.8549576, -71.9400177, 44.0648537, -115.7755280, 115.7949753
34: -60.7338104, 29.7185135, -60.9734192, 30.0123940, -90.7462006, 90.6919327
35: -57.0700150, 34.4891815, -57.2759552, 34.7035294, -91.7735367, 91.7651367
36: -57.1203461, 33.6009216, -57.3456383, 33.9027748, -91.0231171, 90.9465561
37: -84.9560623, 32.7547531, -85.3390045, 33.0833359, -118.0393982, 118.0937576
38: -68.9009705, 40.5982285, -69.1595001, 40.9228325, -109.8237915, 109.7577286
39: -84.8436050, 40.5689850, -85.1009521, 40.7674065, -125.6110077, 125.6699371
40: -75.0684967, 29.8764820, -75.2598038, 30.0212746, -105.0897598, 105.1362839
41: -54.1321602, 25.7044086, -54.3743706, 25.9574394, -80.0895920, 80.0787811
42: -38.8338699, 29.2097206, -38.9792786, 29.4161606, -68.2500305, 68.1889954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=405, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 599
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1491
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 604

## Relational analysis of IS_A1_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.7661816, upper bound: 38.9896128
time: 77.99 seconds

## Relational analysis of IS_A1_A1_A1_A2

### Relational analysis result of IS_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.8042787, upper bound: 38.9925077
time: 78.11 seconds

## BFS IS instance: IS_A1_A1_A2

### Backsubstitution after applying IS history:
0: -53.1766129, 42.9205475, -53.4169159, 43.0670891, -96.2436981, 96.3374634
1: -31.5090427, 35.9571114, -31.6839504, 36.0978775, -67.6069107, 67.6410599
2: -30.2649059, 35.5151520, -30.4485970, 35.6688919, -65.9337921, 65.9637451
3: -33.7639275, 41.4666290, -33.9768639, 41.6637383, -75.4276581, 75.4434967
4: -39.8686676, 38.7936211, -40.0776138, 38.9796715, -78.8483429, 78.8712311
5: -36.7173195, 41.2480049, -36.9238434, 41.4467125, -78.1640320, 78.1718445
6: -55.8787842, 22.3717842, -55.9969635, 22.5004158, -78.3791962, 78.3687439
7: -42.7778397, 40.1101913, -42.9874802, 40.2598953, -83.0377274, 83.0976639
8: -39.2199821, 45.4054794, -39.4278259, 45.5872078, -84.8071899, 84.8333054
9: -34.0172920, 37.4648094, -34.2170639, 37.5508575, -71.5681458, 71.6818695
10: -55.1240158, 52.2095375, -55.2783737, 52.3924789, -107.5164948, 107.4879150
11: -56.3987732, 39.5351906, -56.5973816, 39.7292938, -96.1280670, 96.1325684
12: -59.0225067, 43.8361053, -59.2368927, 44.0629539, -103.0854568, 103.0729980
13: -48.5774307, 49.5233879, -48.7874832, 49.6723518, -98.2497864, 98.3108673
14: -81.4594879, 43.2827568, -81.6445465, 43.4197388, -124.8792114, 124.9273071
15: -40.2578735, 36.3208008, -40.4406052, 36.4350510, -76.6929245, 76.7614059
16: -58.1067467, 40.8379288, -58.3419228, 40.9164429, -99.0231857, 99.1798477
17: -85.1246796, 62.4355659, -85.3112564, 62.5844116, -147.7090759, 147.7468262
18: -48.9461555, 28.9014606, -49.0801201, 29.1333771, -78.0795288, 77.9815826
19: -41.3147049, 19.3691120, -41.4566994, 19.5136356, -60.8283386, 60.8258133
20: -35.3529282, 21.6481743, -35.4687958, 21.8055801, -57.1585083, 57.1169701
21: -49.1220474, 25.3230801, -49.2746315, 25.4733677, -74.5954056, 74.5977020
22: -50.9386520, 29.9402943, -51.0891800, 30.1129665, -81.0516205, 81.0294724
23: -39.0519791, 26.4098816, -39.2448349, 26.6008987, -65.6528778, 65.6547089
24: -45.1624451, 22.7132874, -45.3265572, 22.8662701, -68.0287170, 68.0398407
25: -38.4730301, 30.8819275, -38.6150780, 31.0640774, -69.5371094, 69.4970093
26: -59.0041656, 37.3707886, -59.2024460, 37.6605377, -96.6646881, 96.5732346
27: -49.2999763, 27.1836166, -49.4697571, 27.3583126, -76.6582870, 76.6533661
28: -37.8116379, 28.6693134, -37.9396820, 28.8449516, -66.6565857, 66.6089935
29: -55.4045525, 34.2375984, -55.5710373, 34.3960495, -89.8005981, 89.8086243
30: -47.7329025, 27.1323566, -47.8752861, 27.2696686, -75.0025711, 75.0076447
31: -48.9166832, 23.8725090, -49.1185150, 24.0397968, -72.9564743, 72.9910278
32: -49.0269585, 27.2829399, -49.2172089, 27.4615250, -76.4884796, 76.5001526
33: -71.8153534, 44.0139885, -71.9575806, 44.1206741, -115.9360275, 115.9715652
34: -60.8311386, 29.8529453, -60.9900055, 30.0580177, -90.8891449, 90.8429489
35: -57.1836166, 34.6153870, -57.2929955, 34.7482071, -91.9318237, 91.9083862
36: -57.2010422, 33.7454529, -57.3602486, 33.9534912, -91.1545334, 91.1056976
37: -85.1157913, 32.9731712, -85.3623810, 33.1613083, -118.2770844, 118.3355408
38: -68.9644699, 40.6552277, -69.1776123, 40.9405365, -109.9050064, 109.8328400
39: -84.9271164, 40.6618919, -85.1232300, 40.8005295, -125.7276459, 125.7851181
40: -75.1363983, 29.9503326, -75.2769165, 30.0451355, -105.1815338, 105.2272491
41: -54.2065773, 25.8191185, -54.3854103, 25.9961128, -80.2026825, 80.2045288
42: -38.8735199, 29.2909279, -38.9878349, 29.4422855, -68.3157959, 68.2787628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=405, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 599
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 729

## Relational analysis of IS_A1_A1_A2_B1

### Relational analysis result of IS_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.8770874, upper bound: 38.9244806
time: 91.37 seconds

## Relational analysis of IS_A1_A1_A2_B2

### Relational analysis result of IS_A1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.8770874, upper bound: 38.9244806
time: 124.73 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -53.3558388, 43.0110016, -53.4660034, 43.0752792, -96.4311066, 96.4770050
1: -31.6315670, 36.0353012, -31.7199440, 36.1044846, -67.7360458, 67.7552490
2: -30.3952808, 35.6005096, -30.4876919, 35.6756058, -66.0708847, 66.0882034
3: -33.9039993, 41.5664749, -34.0193176, 41.6733284, -75.5773315, 75.5857925
4: -40.0211792, 38.8927917, -40.1242065, 38.9868851, -79.0080566, 79.0169983
5: -36.8515778, 41.3545532, -36.9648933, 41.4571915, -78.3087616, 78.3194427
6: -55.9416847, 22.5031204, -56.0050850, 22.5385208, -78.4802017, 78.5082016
7: -42.9173431, 40.1884308, -43.0267029, 40.2648735, -83.1822205, 83.2151337
8: -39.3753433, 45.5147133, -39.4727097, 45.5969582, -84.9722977, 84.9874268
9: -34.1565170, 37.5060616, -34.2582703, 37.5594482, -71.7159653, 71.7643280
10: -55.2172623, 52.3386040, -55.2985153, 52.4314461, -107.6487122, 107.6371155
11: -56.5125923, 39.6945953, -56.6117287, 39.7769928, -96.2895813, 96.3063202
12: -59.1524353, 44.0282097, -59.2473412, 44.1215210, -103.2739563, 103.2755508
13: -48.7232018, 49.5968628, -48.8264618, 49.6870956, -98.4102936, 98.4233246
14: -81.5798492, 43.3515396, -81.6702423, 43.4405823, -125.0204315, 125.0217819
15: -40.4161339, 36.3930054, -40.4856415, 36.4333725, -76.8494949, 76.8786469
16: -58.2632599, 40.8810349, -58.3751831, 40.9230385, -99.1862946, 99.2562180
17: -85.2481079, 62.4594879, -85.3441544, 62.5809555, -147.8290710, 147.8036346
18: -49.0330162, 29.0793571, -49.1019135, 29.1873188, -78.2203369, 78.1812744
19: -41.3755798, 19.4622192, -41.4667053, 19.5416355, -60.9172058, 60.9289246
20: -35.4298630, 21.7677670, -35.4753265, 21.8421345, -57.2719955, 57.2430954
21: -49.2043037, 25.4247131, -49.2905579, 25.5039444, -74.7082367, 74.7152710
22: -51.0166283, 30.0596561, -51.1049995, 30.1451035, -81.1617279, 81.1646500
23: -39.1444931, 26.5340519, -39.2553711, 26.6389771, -65.7834702, 65.7894211
24: -45.2383423, 22.8011513, -45.3395157, 22.8918686, -68.1302109, 68.1406631
25: -38.5465851, 30.9937782, -38.6289864, 31.0941982, -69.6407852, 69.6227646
26: -59.1239548, 37.5939445, -59.2188950, 37.7267036, -96.8506622, 96.8128357
27: -49.3938103, 27.3266182, -49.4805031, 27.4031754, -76.7969818, 76.8071213
28: -37.8646584, 28.7835045, -37.9514008, 28.8792648, -66.7439270, 66.7349014
29: -55.4789963, 34.3438835, -55.5867691, 34.4274254, -89.9064178, 89.9306488
30: -47.8174820, 27.2297230, -47.8948021, 27.2970886, -75.1145706, 75.1245270
31: -49.0132599, 23.9846039, -49.1335297, 24.0736771, -73.0869293, 73.1181335
32: -49.1527824, 27.4661331, -49.2245560, 27.5193901, -76.6721725, 76.6906891
33: -71.8627777, 44.0733681, -71.9623337, 44.1330109, -115.9957809, 116.0357056
34: -60.9255219, 30.0089436, -61.0031891, 30.1058617, -91.0313873, 91.0121307
35: -57.2464447, 34.7119904, -57.3030930, 34.7746620, -92.0211029, 92.0150833
36: -57.3195457, 33.9428291, -57.3717384, 34.0171051, -91.3366547, 91.3145676
37: -85.2447968, 33.0904846, -85.3828888, 33.1941986, -118.4389954, 118.4733734
38: -69.1229553, 40.9215050, -69.1963348, 41.0277939, -110.1507416, 110.1178360
39: -85.0538177, 40.7777328, -85.1446457, 40.8302917, -125.8841095, 125.9223785
40: -75.2199020, 30.0305672, -75.2948456, 30.0671730, -105.2870789, 105.3254089
41: -54.3102264, 25.9817333, -54.3959846, 26.0450974, -80.3553162, 80.3777161
42: -38.9370956, 29.4267960, -38.9945221, 29.4824905, -68.4195786, 68.4213181

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 729
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 599
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 1491
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 729

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.8653287, upper bound: 38.9980841
time: 82.34 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.8071526, upper bound: 38.9990716
time: 73.29 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 158.07 seconds
IS_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 158.07
Output dim: 2, lower bound: -38.7661816, upper bound: 38.9896128
IS_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 158.07
Output dim: 2, lower bound: -38.8042787, upper bound: 38.9925077
IS_A1_A1_A2_B1, status: Status.VERIFIED, split count: 4, time: 158.07
Output dim: 2, lower bound: -38.8770874, upper bound: 38.9244806
IS_A1_A1_A2_B2, status: Status.VERIFIED, split count: 4, time: 158.07
Output dim: 2, lower bound: -38.8770874, upper bound: 38.9244806
IS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 158.07
Output dim: 2, lower bound: -38.8653287, upper bound: 38.9980841
IS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 158.07
Output dim: 2, lower bound: -38.8071526, upper bound: 38.9990716

## BFS IS instance: IS_A1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -52.8358002, 42.7701263, -53.3092117, 43.0514832, -95.8872833, 96.0793381
1: -31.2798901, 35.8422279, -31.6054077, 36.0840225, -67.3639145, 67.4476318
2: -30.0042114, 35.3685150, -30.3592968, 35.6554108, -65.6596222, 65.7278137
3: -33.4832230, 41.2860641, -33.8780403, 41.6452827, -75.1285019, 75.1641006
4: -39.6207695, 38.6486816, -39.9934769, 38.9645882, -78.5853500, 78.6421585
5: -36.4467621, 41.0710449, -36.8294144, 41.4282303, -77.8749847, 77.9004593
6: -55.7761078, 22.2349339, -55.9751205, 22.4481697, -78.2242737, 78.2100525
7: -42.4704018, 39.9418869, -42.8781548, 40.2401428, -82.7105408, 82.8200378
8: -38.8934860, 45.1999359, -39.3138161, 45.5659256, -84.4594116, 84.5137482
9: -33.8270760, 37.3718491, -34.1540909, 37.5328751, -71.3599548, 71.5259399
10: -54.9660225, 52.0718155, -55.2308273, 52.3515892, -107.3176117, 107.3026428
11: -56.2219658, 39.3385391, -56.5629311, 39.6620102, -95.8839722, 95.9014740
12: -58.8622513, 43.5613556, -59.2172699, 43.9715691, -102.8338165, 102.7786255
13: -48.3646278, 49.3462257, -48.7126923, 49.6392250, -98.0038528, 98.0589142
14: -81.1414337, 43.1021957, -81.5421448, 43.3852768, -124.5267029, 124.6443405
15: -40.0624008, 36.2414017, -40.3679352, 36.4153481, -76.4777451, 76.6093369
16: -57.9314613, 40.7627335, -58.2911034, 40.8890381, -98.8204956, 99.0538330
17: -84.9313278, 62.2913933, -85.2533951, 62.5379219, -147.4692535, 147.5447845
18: -48.7777061, 28.6758156, -49.0492668, 29.0559883, -77.8336945, 77.7250748
19: -41.1636162, 19.1932564, -41.4347687, 19.4520054, -60.6156235, 60.6280174
20: -35.2461853, 21.5074654, -35.4443207, 21.7566147, -57.0027924, 56.9517746
21: -48.9624100, 25.1495094, -49.2470207, 25.4146385, -74.3770447, 74.3965302
22: -50.7435684, 29.6915092, -51.0493660, 30.0248528, -80.7684174, 80.7408752
23: -38.8821945, 26.2224159, -39.2222366, 26.5368481, -65.4190369, 65.4446564
24: -44.9496346, 22.5201588, -45.2931442, 22.7993031, -67.7489395, 67.8133011
25: -38.2917252, 30.6312542, -38.5847702, 30.9762859, -69.2680054, 69.2160187
26: -58.7686462, 36.9847679, -59.1666222, 37.5264053, -96.2950516, 96.1513901
27: -49.1324730, 27.0190830, -49.4383278, 27.2978096, -76.4302750, 76.4574051
28: -37.6558495, 28.4770660, -37.9194183, 28.7805290, -66.4363785, 66.3964844
29: -55.1761475, 34.0145760, -55.5290565, 34.3170242, -89.4931717, 89.5436325
30: -47.5394897, 26.9498081, -47.8420868, 27.2097588, -74.7492523, 74.7918854
31: -48.7152176, 23.7031174, -49.0875435, 23.9821568, -72.6973724, 72.7906570
32: -48.9189873, 27.1608810, -49.1936607, 27.4123669, -76.3313446, 76.3545380
33: -71.6457291, 43.8251228, -71.9216461, 44.0546608, -115.7003937, 115.7467651
34: -60.6845360, 29.6326065, -60.9629745, 29.9844208, -90.6689606, 90.5955811
35: -57.0234070, 34.4503670, -57.2632523, 34.6917114, -91.7151184, 91.7136230
36: -57.0754318, 33.5293236, -57.3364410, 33.8786812, -90.9541168, 90.8657532
37: -84.8505020, 32.6638947, -85.3220520, 33.0531693, -117.9036713, 117.9859467
38: -68.8321838, 40.4916153, -69.1463165, 40.8886261, -109.7208099, 109.6379242
39: -84.7676697, 40.5390205, -85.0818024, 40.7571526, -125.5248260, 125.6208191
40: -74.9949341, 29.8320332, -75.2447662, 30.0065613, -105.0014954, 105.0767975
41: -54.0604820, 25.6172829, -54.3638954, 25.9288349, -79.9893188, 79.9811783
42: -38.7896461, 29.1311913, -38.9719086, 29.3905754, -68.1802216, 68.1031036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=207, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=404, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 599
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1491
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 976

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 605

## Relational analysis of IS_A1_A1_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.7304934, upper bound: 38.9868101
time: 78.23 seconds

## Relational analysis of IS_A1_A1_A1_A1_A2

### Relational analysis result of IS_A1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.7304934, upper bound: 38.9883318
time: 85.36 seconds

## BFS IS instance: IS_A1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -52.9575958, 42.8354759, -53.3460312, 43.0565834, -96.0141754, 96.1814957
1: -31.3518581, 35.8860970, -31.6292419, 36.0884895, -67.4403381, 67.5153351
2: -30.1052914, 35.4331856, -30.3931637, 35.6598129, -65.7651062, 65.8263474
3: -33.6029053, 41.3807983, -33.9192238, 41.6520157, -75.2549133, 75.3000183
4: -39.7275620, 38.7140427, -40.0285873, 38.9698563, -78.6974182, 78.7426300
5: -36.5606766, 41.1593094, -36.8682976, 41.4352341, -77.9959106, 78.0276031
6: -55.8250732, 22.3000183, -55.9839516, 22.4696388, -78.2947083, 78.2839661
7: -42.5624504, 39.9945145, -42.9098854, 40.2447586, -82.8072052, 82.9044037
8: -38.9889297, 45.2643356, -39.3451462, 45.5715866, -84.5605087, 84.6094742
9: -33.9273872, 37.4102936, -34.1861343, 37.5383911, -71.4657669, 71.5964279
10: -55.0222168, 52.1480713, -55.2454338, 52.3746185, -107.3968353, 107.3935013
11: -56.2943192, 39.4483566, -56.5714378, 39.6998024, -95.9941254, 96.0197906
12: -58.9389648, 43.6689453, -59.2240753, 44.0063324, -102.9452972, 102.8930206
13: -48.5000076, 49.4450226, -48.7586136, 49.6509438, -98.1509552, 98.2036362
14: -81.2076645, 43.1701431, -81.5590668, 43.4090347, -124.6166992, 124.7292099
15: -40.1498871, 36.2770729, -40.3962326, 36.4218292, -76.5717087, 76.6733093
16: -58.0081520, 40.7668610, -58.3121262, 40.8888245, -98.8969727, 99.0789795
17: -84.9904099, 62.3320503, -85.2707520, 62.5500336, -147.5404358, 147.6027985
18: -48.8477516, 28.8131104, -49.0618935, 29.1022224, -77.9499741, 77.8750000
19: -41.2341995, 19.2811546, -41.4424667, 19.4822845, -60.7164803, 60.7236214
20: -35.3020020, 21.5925560, -35.4527512, 21.7847252, -57.0867271, 57.0453033
21: -49.0387230, 25.2458038, -49.2570267, 25.4474373, -74.4861603, 74.5028229
22: -50.8111076, 29.7767143, -51.0591316, 30.0538101, -80.8649139, 80.8358459
23: -38.9586830, 26.3215466, -39.2291794, 26.5705338, -65.5292206, 65.5507202
24: -45.0366135, 22.6174755, -45.3013687, 22.8322277, -67.8688354, 67.9188385
25: -38.3599319, 30.7298546, -38.5928802, 31.0097637, -69.3696899, 69.3227310
26: -58.8604355, 37.1505356, -59.1765862, 37.5815430, -96.4419708, 96.3271179
27: -49.2319756, 27.1424465, -49.4499817, 27.3397579, -76.5717316, 76.5924301
28: -37.7262764, 28.5764236, -37.9267578, 28.8135567, -66.5398331, 66.5031815
29: -55.2571297, 34.1014061, -55.5390396, 34.3474464, -89.6045761, 89.6404419
30: -47.6115036, 27.0420055, -47.8513756, 27.2403049, -74.8518066, 74.8933792
31: -48.8147888, 23.8014755, -49.0984993, 24.0155525, -72.8303375, 72.8999634
32: -48.9800491, 27.2219391, -49.2029724, 27.4325619, -76.4126053, 76.4249115
33: -71.6828766, 43.8515167, -71.9311981, 44.0637283, -115.7465897, 115.7827148
34: -60.7303848, 29.7137985, -60.9723473, 30.0108509, -90.7412338, 90.6861420
35: -57.0550880, 34.4858932, -57.2710876, 34.7024536, -91.7575378, 91.7569733
36: -57.1154213, 33.5971642, -57.3440742, 33.9016113, -91.0170288, 90.9412384
37: -84.9506531, 32.7502136, -85.3372803, 33.0819321, -118.0325851, 118.0874939
38: -68.8952332, 40.5922470, -69.1576385, 40.9208984, -109.8161163, 109.7498856
39: -84.8347397, 40.5648041, -85.0980225, 40.7660599, -125.6007996, 125.6628189
40: -75.0635834, 29.8738632, -75.2582855, 30.0204620, -105.0840454, 105.1321411
41: -54.1273842, 25.7001343, -54.3729134, 25.9560299, -80.0834045, 80.0730438
42: -38.8301125, 29.2054901, -38.9780731, 29.4147835, -68.2448959, 68.1835632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=207, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=405, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 599
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1491
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 605

## Relational analysis of IS_A1_A1_A1_A2_B1

### Relational analysis result of IS_A1_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.7617056, upper bound: 38.9506517
time: 70.50 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2

### Relational analysis result of IS_A1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.8031705, upper bound: 38.9914289
time: 78.46 seconds

## BFS IS instance: IS_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -53.1378365, 42.9259872, -53.4660034, 43.0752792, -96.2131195, 96.3919830
1: -31.4751740, 35.9641724, -31.7199440, 36.1044846, -67.5796585, 67.6841125
2: -30.2364674, 35.5186729, -30.4876919, 35.6756058, -65.9120712, 66.0063629
3: -33.7447662, 41.4814301, -34.0193176, 41.6733284, -75.4180908, 75.5007477
4: -39.8809662, 38.8138428, -40.1242065, 38.9868851, -78.8678436, 78.9380493
5: -36.6966324, 41.2660751, -36.9648933, 41.4571915, -78.1538086, 78.2309723
6: -55.8893852, 22.4334087, -56.0050850, 22.5385208, -78.4279022, 78.4384918
7: -42.7013588, 40.0706329, -43.0267029, 40.2648735, -82.9662170, 83.0973282
8: -39.1431656, 45.3756638, -39.4727097, 45.5969582, -84.7401199, 84.8483734
9: -34.0690613, 37.4522247, -34.2582703, 37.5594482, -71.6285095, 71.7104950
10: -55.1169739, 52.2800827, -55.2985153, 52.4314461, -107.5484161, 107.5785828
11: -56.4091949, 39.6087799, -56.6117287, 39.7769928, -96.1861801, 96.2205048
12: -59.0694504, 43.8627777, -59.2473412, 44.1215210, -103.1909637, 103.1101151
13: -48.6523972, 49.5179939, -48.8264618, 49.6870956, -98.3394928, 98.3444519
14: -81.3251953, 43.2439270, -81.6702423, 43.4405823, -124.7657776, 124.9141693
15: -40.3079453, 36.3375397, -40.4856415, 36.4333725, -76.7413101, 76.8231735
16: -58.1607590, 40.8259735, -58.3751831, 40.9230385, -99.0837936, 99.2011566
17: -85.1154404, 62.3742523, -85.3441544, 62.5809555, -147.6963806, 147.7183990
18: -48.9370842, 28.9951286, -49.1019135, 29.1873188, -78.1244049, 78.0970459
19: -41.2970963, 19.3756371, -41.4667053, 19.5416355, -60.8387299, 60.8423424
20: -35.3741646, 21.7151814, -35.4753265, 21.8421345, -57.2163010, 57.1904984
21: -49.1222000, 25.3494434, -49.2905579, 25.5039444, -74.6261444, 74.6399994
22: -50.8882179, 29.8959465, -51.1049995, 30.1451035, -81.0333252, 81.0009460
23: -39.0526886, 26.4478207, -39.2553711, 26.6389771, -65.6916656, 65.7031937
24: -45.1138458, 22.7076035, -45.3395157, 22.8918686, -68.0057144, 68.0471191
25: -38.4334030, 30.8421593, -38.6289864, 31.0941982, -69.5276031, 69.4711456
26: -58.9817619, 37.3763504, -59.2188950, 37.7267036, -96.7084656, 96.5952454
27: -49.3223419, 27.2893715, -49.4805031, 27.4031754, -76.7255173, 76.7698746
28: -37.7808685, 28.6927452, -37.9514008, 28.8792648, -66.6601334, 66.6441422
29: -55.3324280, 34.2097092, -55.5867691, 34.4274254, -89.7598419, 89.7964783
30: -47.6971359, 27.1411572, -47.8948021, 27.2970886, -74.9942245, 75.0359573
31: -48.9129601, 23.9158535, -49.1335297, 24.0736771, -72.9866333, 73.0493851
32: -49.1023254, 27.4072914, -49.2245560, 27.5193901, -76.6217194, 76.6318512
33: -71.7546768, 43.9097366, -71.9623337, 44.1330109, -115.8876877, 115.8720703
34: -60.8252258, 29.8712196, -61.0031891, 30.1058617, -90.9310760, 90.8744049
35: -57.1293640, 34.5823364, -57.3030930, 34.7746620, -91.9040222, 91.8854294
36: -57.2361679, 33.7950211, -57.3717384, 34.0171051, -91.2532654, 91.1667633
37: -85.0817184, 32.8667030, -85.3828888, 33.1941986, -118.2759171, 118.2495880
38: -69.0565643, 40.8627396, -69.1963348, 41.0277939, -110.0843353, 110.0590744
39: -84.9665527, 40.6769371, -85.1446457, 40.8302917, -125.7968445, 125.8215790
40: -75.1488495, 29.9533806, -75.2948456, 30.0671730, -105.2160187, 105.2482300
41: -54.2333374, 25.8628883, -54.3959846, 26.0450974, -80.2784348, 80.2588730
42: -38.8951988, 29.3420792, -38.9945221, 29.4824905, -68.3776855, 68.3366013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=210, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 599
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1491
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 604

## Relational analysis of IS_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.8601887, upper bound: 38.9412397
time: 85.68 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.8639624, upper bound: 38.9967232
time: 78.95 seconds

## BFS IS instance: IS_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -53.3501663, 43.0084381, -53.4660034, 43.0752792, -96.4254456, 96.4744415
1: -31.6278381, 36.0333214, -31.7199440, 36.1044846, -67.7323227, 67.7532654
2: -30.3915405, 35.5982780, -30.4876919, 35.6756058, -66.0671387, 66.0859680
3: -33.9000244, 41.5645180, -34.0193176, 41.6733284, -75.5733490, 75.5838318
4: -40.0175171, 38.8905869, -40.1242065, 38.9868851, -79.0043945, 79.0147934
5: -36.8478546, 41.3524323, -36.9648933, 41.4571915, -78.3050461, 78.3173218
6: -55.9383621, 22.4989777, -56.0050850, 22.5385208, -78.4768753, 78.5040588
7: -42.9113846, 40.1843796, -43.0267029, 40.2648735, -83.1762543, 83.2110825
8: -39.3696823, 45.5123749, -39.4727097, 45.5969582, -84.9666367, 84.9850845
9: -34.1537018, 37.5042114, -34.2582703, 37.5594482, -71.7131500, 71.7624817
10: -55.2137833, 52.3365326, -55.2985153, 52.4314461, -107.6452332, 107.6350403
11: -56.5082169, 39.6899338, -56.6117287, 39.7769928, -96.2852097, 96.3016663
12: -59.1496201, 44.0242462, -59.2473412, 44.1215210, -103.2711334, 103.2715912
13: -48.7198868, 49.5918388, -48.8264618, 49.6870956, -98.4069824, 98.4182968
14: -81.5727539, 43.3501968, -81.6702423, 43.4405823, -125.0133286, 125.0204391
15: -40.4110107, 36.3789940, -40.4856415, 36.4333725, -76.8443756, 76.8646317
16: -58.2533340, 40.8778305, -58.3751831, 40.9230385, -99.1763687, 99.2530136
17: -85.2439575, 62.4524078, -85.3441544, 62.5809555, -147.8248901, 147.7965698
18: -49.0309258, 29.0762558, -49.1019135, 29.1873188, -78.2182465, 78.1781616
19: -41.3738518, 19.4596329, -41.4667053, 19.5416355, -60.9154892, 60.9263382
20: -35.4225235, 21.7656746, -35.4753265, 21.8421345, -57.2646561, 57.2409973
21: -49.2019730, 25.4223366, -49.2905579, 25.5039444, -74.7059174, 74.7128906
22: -51.0117989, 30.0544662, -51.1049995, 30.1451035, -81.1568985, 81.1594620
23: -39.1427956, 26.5313358, -39.2553711, 26.6389771, -65.7817688, 65.7867050
24: -45.2352333, 22.7985229, -45.3395157, 22.8918686, -68.1270981, 68.1380310
25: -38.5427208, 30.9889812, -38.6289864, 31.0941982, -69.6369171, 69.6179657
26: -59.1207771, 37.5885010, -59.2188950, 37.7267036, -96.8474731, 96.8073959
27: -49.3852654, 27.3247089, -49.4805031, 27.4031754, -76.7884369, 76.8052063
28: -37.8632698, 28.7810192, -37.9514008, 28.8792648, -66.7425308, 66.7324066
29: -55.4745712, 34.3401604, -55.5867691, 34.4274254, -89.9019928, 89.9269257
30: -47.8139038, 27.2268410, -47.8948021, 27.2970886, -75.1109848, 75.1216431
31: -49.0107803, 23.9823112, -49.1335297, 24.0736771, -73.0844574, 73.1158371
32: -49.1450157, 27.4631748, -49.2245560, 27.5193901, -76.6643982, 76.6877289
33: -71.8593292, 44.0688515, -71.9623337, 44.1330109, -115.9923325, 116.0311813
34: -60.9225082, 30.0056572, -61.0031891, 30.1058617, -91.0283661, 91.0088501
35: -57.2429657, 34.7085915, -57.3030930, 34.7746620, -92.0176239, 92.0116882
36: -57.3167648, 33.9395599, -57.3717384, 34.0171051, -91.3338699, 91.3112946
37: -85.2414703, 33.0850830, -85.3828888, 33.1941986, -118.4356689, 118.4679718
38: -69.1200714, 40.9197311, -69.1963348, 41.0277939, -110.1478424, 110.1160660
39: -85.0500565, 40.7699051, -85.1446457, 40.8302917, -125.8803406, 125.9145508
40: -75.2168198, 30.0272446, -75.2948456, 30.0671730, -105.2839966, 105.3220825
41: -54.3077087, 25.9775810, -54.3959846, 26.0450974, -80.3528061, 80.3735657
42: -38.9348373, 29.4232883, -38.9945221, 29.4824905, -68.4173279, 68.4178085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 599
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 604

## Relational analysis of IS_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.8601887, upper bound: 38.8747330
time: 72.55 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.8639624, upper bound: 38.9302629
time: 73.48 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 148.47 seconds
IS_A1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 148.47
Output dim: 2, lower bound: -38.7304934, upper bound: 38.9868101
IS_A1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 148.47
Output dim: 2, lower bound: -38.7304934, upper bound: 38.9883318
IS_A1_A1_A1_A2_B1, status: Status.VERIFIED, split count: 5, time: 148.47
Output dim: 2, lower bound: -38.7617056, upper bound: 38.9506517
IS_A1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 148.47
Output dim: 2, lower bound: -38.8031705, upper bound: 38.9914289
IS_A1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 148.47
Output dim: 2, lower bound: -38.8601887, upper bound: 38.9412397
IS_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 148.47
Output dim: 2, lower bound: -38.8639624, upper bound: 38.9967232
IS_A1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 148.47
Output dim: 2, lower bound: -38.8601887, upper bound: 38.8747330
IS_A1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 148.47
Output dim: 2, lower bound: -38.8639624, upper bound: 38.9302629

## BFS IS instance: IS_A1_A1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -52.7303162, 42.7275848, -53.2823067, 43.0450249, -95.7753296, 96.0098877
1: -31.2215118, 35.8101730, -31.5882244, 36.0769920, -67.2985077, 67.3983994
2: -29.9019794, 35.2935905, -30.3274155, 35.6482925, -65.5502701, 65.6210022
3: -33.3575592, 41.1727448, -33.8374786, 41.6357536, -74.9933014, 75.0102234
4: -39.5148773, 38.5794601, -39.9612045, 38.9563675, -78.4712372, 78.5406647
5: -36.3210602, 40.9651489, -36.7888412, 41.4183350, -77.7393951, 77.7539825
6: -55.7033501, 22.1851807, -55.9570732, 22.4340134, -78.1373596, 78.1422577
7: -42.3777847, 39.8956909, -42.8487167, 40.2330589, -82.6108398, 82.7444077
8: -38.8201218, 45.1350784, -39.2927322, 45.5530586, -84.3731689, 84.4278030
9: -33.7472229, 37.3434296, -34.1315994, 37.5262909, -71.2735138, 71.4750214
10: -54.8604736, 51.9183273, -55.2149658, 52.3039551, -107.1644287, 107.1332932
11: -56.1090508, 39.2033424, -56.5519104, 39.6179199, -95.7269745, 95.7552490
12: -58.7678452, 43.4574242, -59.2052536, 43.9397621, -102.7076035, 102.6626663
13: -48.1471519, 49.1689758, -48.6425323, 49.6257248, -97.7728729, 97.8114929
14: -81.0031204, 42.9804993, -81.5211334, 43.3459549, -124.3490677, 124.5016327
15: -39.9943542, 36.1962967, -40.3481407, 36.4050255, -76.3993759, 76.5444336
16: -57.8470306, 40.7165756, -58.2688026, 40.8742599, -98.7212906, 98.9853745
17: -84.8631516, 62.1994438, -85.2357178, 62.5133934, -147.3765411, 147.4351501
18: -48.6661644, 28.5098610, -49.0345955, 29.0030556, -77.6692200, 77.5444565
19: -41.0779343, 19.1091671, -41.4246979, 19.4239349, -60.5018692, 60.5338669
20: -35.1833115, 21.4210186, -35.4348412, 21.7305813, -56.9138908, 56.8558578
21: -48.8653374, 25.0495853, -49.2344398, 25.3821297, -74.2474670, 74.2840271
22: -50.6522064, 29.5975876, -51.0363464, 29.9949665, -80.6471710, 80.6339340
23: -38.7739487, 26.1017189, -39.2130051, 26.4979057, -65.2718506, 65.3147278
24: -44.8311653, 22.4100609, -45.2817841, 22.7643738, -67.5955353, 67.6918488
25: -38.2053833, 30.5284538, -38.5739822, 30.9439659, -69.1493454, 69.1024323
26: -58.6527252, 36.7957954, -59.1547089, 37.4674225, -96.1201324, 95.9504929
27: -48.9887581, 26.8718472, -49.4235191, 27.2496052, -76.2383652, 76.2953644
28: -37.5655060, 28.3658295, -37.9104843, 28.7463799, -66.3118744, 66.2763138
29: -55.0607986, 33.8986511, -55.5155792, 34.2798462, -89.3406372, 89.4142303
30: -47.4463692, 26.8492851, -47.8309441, 27.1781235, -74.6244812, 74.6802216
31: -48.5851898, 23.6102028, -49.0723381, 23.9516525, -72.5368423, 72.6825409
32: -48.8361588, 27.1136398, -49.1789742, 27.3975716, -76.2337341, 76.2926178
33: -71.4930191, 43.7603378, -71.8738098, 44.0443840, -115.5373993, 115.6341400
34: -60.6226158, 29.5431480, -60.9486084, 29.9583263, -90.5809402, 90.4917603
35: -56.8969765, 34.3916855, -57.2249107, 34.6816559, -91.5786285, 91.6165924
36: -56.9804153, 33.5014763, -57.3097496, 33.8709641, -90.8513641, 90.8112259
37: -84.7524567, 32.6063881, -85.2996521, 33.0370941, -117.7895508, 117.9060364
38: -68.7324066, 40.4185524, -69.1183929, 40.8703613, -109.6027679, 109.5369415
39: -84.6011276, 40.4915695, -85.0367966, 40.7504120, -125.3515320, 125.5283661
40: -74.9168091, 29.7932930, -75.2268600, 29.9963169, -104.9131241, 105.0201492
41: -53.9823647, 25.5388031, -54.3499718, 25.9052887, -79.8876495, 79.8887711
42: -38.7302704, 29.0427704, -38.9631462, 29.3636398, -68.0939026, 68.0059128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=404, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 599
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 976

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 728

## Relational analysis of IS_A1_A1_A1_A1_A1_B1

### Relational analysis result of IS_A1_A1_A1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.7265986, upper bound: 38.9172982
time: 79.04 seconds

## Relational analysis of IS_A1_A1_A1_A1_A1_B2

### Relational analysis result of IS_A1_A1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.7265986, upper bound: 38.9857964
time: 80.73 seconds

## BFS IS instance: IS_A1_A1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -52.8283844, 42.7656784, -53.3066177, 43.0499687, -95.8783417, 96.0722885
1: -31.2751522, 35.8361588, -31.6037235, 36.0818825, -67.3570251, 67.4398804
2: -29.9985638, 35.3654213, -30.3573513, 35.6543388, -65.6529007, 65.7227707
3: -33.4759369, 41.2824478, -33.8755302, 41.6440392, -75.1199646, 75.1579742
4: -39.6149292, 38.6441994, -39.9914474, 38.9630470, -78.5779724, 78.6356506
5: -36.4392700, 41.0679016, -36.8267822, 41.4271507, -77.8664169, 77.8946838
6: -55.7644081, 22.2261486, -55.9707794, 22.4450645, -78.2094727, 78.1969299
7: -42.4630203, 39.9389839, -42.8755264, 40.2391281, -82.7021484, 82.8145065
8: -38.8852386, 45.1943512, -39.3109512, 45.5639534, -84.4491882, 84.5052948
9: -33.8211517, 37.3616066, -34.1520195, 37.5293312, -71.3504791, 71.5136261
10: -54.9615517, 52.0620003, -55.2292709, 52.3482132, -107.3097610, 107.2912598
11: -56.2138596, 39.3316116, -56.5601273, 39.6597099, -95.8735580, 95.8917389
12: -58.8574448, 43.5530128, -59.2155762, 43.9687233, -102.8261719, 102.7685852
13: -48.3519936, 49.3386612, -48.7083130, 49.6366386, -97.9886322, 98.0469742
14: -81.1334076, 43.0951691, -81.5393524, 43.3828812, -124.5162888, 124.6345062
15: -40.0526810, 36.2354584, -40.3645630, 36.4132385, -76.4659195, 76.6000214
16: -57.9230003, 40.7287750, -58.2881279, 40.8777084, -98.8007050, 99.0169067
17: -84.9227753, 62.2747269, -85.2503662, 62.5315094, -147.4542847, 147.5250854
18: -48.7698898, 28.6665611, -49.0465279, 29.0528183, -77.8227081, 77.7130890
19: -41.1592026, 19.1876221, -41.4331970, 19.4498959, -60.6090927, 60.6208153
20: -35.2411880, 21.5023918, -35.4425850, 21.7548428, -56.9960327, 56.9449768
21: -48.9558372, 25.1437340, -49.2446938, 25.4126778, -74.3685150, 74.3884277
22: -50.7380867, 29.6855431, -51.0474052, 30.0228271, -80.7609100, 80.7329407
23: -38.8770599, 26.2157631, -39.2204742, 26.5345383, -65.4115982, 65.4362335
24: -44.9431915, 22.5141487, -45.2909203, 22.7972050, -67.7403946, 67.8050690
25: -38.2869110, 30.6250343, -38.5830803, 30.9741192, -69.2610321, 69.2081146
26: -58.7608604, 36.9743004, -59.1639252, 37.5227966, -96.2836609, 96.1382217
27: -49.1241531, 27.0118027, -49.4354630, 27.2953415, -76.4194946, 76.4472656
28: -37.6503830, 28.4713383, -37.9175262, 28.7785225, -66.4289093, 66.3888550
29: -55.1694374, 34.0083923, -55.5266685, 34.3149376, -89.4843674, 89.5350571
30: -47.5287628, 26.9438972, -47.8383408, 27.2077274, -74.7364883, 74.7822342
31: -48.7099571, 23.6975899, -49.0857162, 23.9802780, -72.6902313, 72.7833099
32: -48.9139595, 27.1545391, -49.1919174, 27.4101772, -76.3241272, 76.3464584
33: -71.6274719, 43.8209991, -71.9149933, 44.0531540, -115.6806259, 115.7359924
34: -60.6777496, 29.6271400, -60.9605103, 29.9824696, -90.6602173, 90.5876465
35: -57.0070152, 34.4466324, -57.2578468, 34.6904068, -91.6974182, 91.7044754
36: -57.0613480, 33.5264969, -57.3307800, 33.8776436, -90.9389877, 90.8572769
37: -84.8431396, 32.6548462, -85.3195190, 33.0493851, -117.8925247, 117.9743652
38: -68.8146133, 40.4824867, -69.1400146, 40.8854904, -109.7001038, 109.6224899
39: -84.7475281, 40.5348206, -85.0749588, 40.7556381, -125.5031586, 125.6097794
40: -74.9876022, 29.8263779, -75.2422180, 30.0046463, -104.9922485, 105.0685959
41: -54.0548401, 25.6080151, -54.3619347, 25.9250336, -79.9798737, 79.9699478
42: -38.7851906, 29.1231117, -38.9703560, 29.3876915, -68.1728821, 68.0934677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=404, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1491
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 645
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 976

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 728

## Relational analysis of IS_A1_A1_A1_A1_A2_B1

### Relational analysis result of IS_A1_A1_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.7597084, upper bound: 38.9188074
time: 68.78 seconds

## Relational analysis of IS_A1_A1_A1_A1_A2_B2

### Relational analysis result of IS_A1_A1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.7265986, upper bound: 38.9873174
time: 72.68 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -52.9550476, 42.8340149, -53.3386078, 43.0522308, -96.0072708, 96.1726074
1: -31.3501778, 35.8840218, -31.6245155, 36.0822868, -67.4324646, 67.5085373
2: -30.1033573, 35.4321747, -30.3875103, 35.6568222, -65.7601776, 65.8196869
3: -33.6004410, 41.3795776, -33.9119949, 41.6484222, -75.2488632, 75.2915649
4: -39.7255630, 38.7125702, -40.0227585, 38.9655190, -78.6910858, 78.7353210
5: -36.5580902, 41.1582565, -36.8607407, 41.4321175, -77.9902039, 78.0189972
6: -55.8208961, 22.2969570, -55.9712906, 22.4608364, -78.2817307, 78.2682495
7: -42.5598907, 39.9935303, -42.9025307, 40.2418747, -82.8017654, 82.8960571
8: -38.9860992, 45.2624969, -39.3368607, 45.5661774, -84.5522614, 84.5993500
9: -33.9253273, 37.4068527, -34.1802597, 37.5279007, -71.4532318, 71.5871048
10: -55.0206985, 52.1447906, -55.2409248, 52.3648415, -107.3855286, 107.3857117
11: -56.2916527, 39.4460526, -56.5634880, 39.6928864, -95.9845428, 96.0095367
12: -58.9374008, 43.6661606, -59.2194443, 43.9980545, -102.9354553, 102.8856049
13: -48.4958076, 49.4425316, -48.7461929, 49.6434822, -98.1392899, 98.1887207
14: -81.2049713, 43.1678467, -81.5509644, 43.4018745, -124.6068420, 124.7188034
15: -40.1465607, 36.2750511, -40.3864288, 36.4156799, -76.5622406, 76.6614761
16: -58.0052681, 40.7556000, -58.3037033, 40.8548965, -98.8601685, 99.0593033
17: -84.9874802, 62.3265076, -85.2621460, 62.5343552, -147.5218201, 147.5886536
18: -48.8451271, 28.8100281, -49.0542297, 29.0929928, -77.9381180, 77.8642578
19: -41.2327156, 19.2790661, -41.4380188, 19.4766846, -60.7094002, 60.7170792
20: -35.3003311, 21.5908318, -35.4477921, 21.7796059, -57.0799332, 57.0386200
21: -49.0365295, 25.2438545, -49.2505455, 25.4416084, -74.4781342, 74.4944000
22: -50.8092422, 29.7747536, -51.0536880, 30.0478344, -80.8570786, 80.8284454
23: -38.9569893, 26.3192787, -39.2242050, 26.5638695, -65.5208435, 65.5434799
24: -45.0344658, 22.6154137, -45.2950363, 22.8262386, -67.8607025, 67.9104462
25: -38.3583336, 30.7277184, -38.5880814, 31.0035095, -69.3618469, 69.3157959
26: -58.8578377, 37.1469803, -59.1691017, 37.5711174, -96.4289398, 96.3160858
27: -49.2292557, 27.1400204, -49.4419861, 27.3324795, -76.5617371, 76.5820007
28: -37.7244720, 28.5744476, -37.9213257, 28.8078728, -66.5323486, 66.4957733
29: -55.2548752, 34.0993500, -55.5324402, 34.3412781, -89.5961533, 89.6317902
30: -47.6079865, 27.0400238, -47.8408852, 27.2343960, -74.8423843, 74.8809052
31: -48.8130226, 23.7996101, -49.0931892, 24.0099964, -72.8230133, 72.8927994
32: -48.9783897, 27.2198086, -49.1980705, 27.4262714, -76.4046478, 76.4178772
33: -71.6771774, 43.8500977, -71.9149628, 44.0595131, -115.7366943, 115.7650528
34: -60.7280807, 29.7119083, -60.9653244, 30.0053444, -90.7334290, 90.6772308
35: -57.0497971, 34.4846115, -57.2553215, 34.6987000, -91.7484970, 91.7399292
36: -57.1099243, 33.5961456, -57.3287544, 33.8986969, -91.0086212, 90.9248962
37: -84.9482193, 32.7464447, -85.3298721, 33.0728416, -118.0210571, 118.0763168
38: -68.8891144, 40.5891457, -69.1396332, 40.9117241, -109.8008347, 109.7287750
39: -84.8281250, 40.5633507, -85.0778198, 40.7618141, -125.5899353, 125.6411667
40: -75.0611572, 29.8719807, -75.2509766, 30.0148010, -105.0759583, 105.1229553
41: -54.1254959, 25.6963692, -54.3674240, 25.9468079, -80.0722961, 80.0637894
42: -38.8286362, 29.2026749, -38.9737320, 29.4067955, -68.2354279, 68.1764069

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=207, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=405, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1491
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 728

## Relational analysis of IS_A1_A1_A1_A2_B2_B1

### Relational analysis result of IS_A1_A1_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.7991873, upper bound: 38.9218540
time: 91.11 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_B2

### Relational analysis result of IS_A1_A1_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.8007729, upper bound: 38.8809956
time: 157.87 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -53.1357422, 42.9252014, -53.4594727, 43.0727196, -96.2084351, 96.3846741
1: -31.4737568, 35.9636078, -31.7154217, 36.1026535, -67.5764084, 67.6790314
2: -30.2350559, 35.5179405, -30.4832287, 35.6733208, -65.9083710, 66.0011673
3: -33.7429695, 41.4805679, -34.0136719, 41.6706009, -75.4135666, 75.4942398
4: -39.8795128, 38.8129654, -40.1195831, 38.9842033, -78.8637161, 78.9325485
5: -36.6949158, 41.2653580, -36.9594269, 41.4549255, -78.1498413, 78.2247849
6: -55.8879051, 22.4316635, -56.0004768, 22.5329819, -78.4208832, 78.4321289
7: -42.6996880, 40.0700531, -43.0213928, 40.2630806, -82.9627686, 83.0914459
8: -39.1416931, 45.3743324, -39.4680176, 45.5927124, -84.7344055, 84.8423462
9: -34.0674133, 37.4514313, -34.2531509, 37.5569992, -71.6244049, 71.7045822
10: -55.1153793, 52.2785072, -55.2934074, 52.4265785, -107.5419464, 107.5719147
11: -56.4074936, 39.6072350, -56.6064453, 39.7720413, -96.1795349, 96.2136841
12: -59.0684280, 43.8609848, -59.2441788, 44.1158829, -103.1843109, 103.1051483
13: -48.6492043, 49.5165977, -48.8164558, 49.6826782, -98.3318710, 98.3330536
14: -81.3237305, 43.2424622, -81.6656799, 43.4359894, -124.7597198, 124.9081345
15: -40.3063507, 36.3368225, -40.4806061, 36.4311180, -76.7374573, 76.8174286
16: -58.1588860, 40.8200150, -58.3692055, 40.9034805, -99.0623627, 99.1892166
17: -85.1135635, 62.3671188, -85.3383331, 62.5584373, -147.6719971, 147.7054443
18: -48.9356728, 28.9928608, -49.0974045, 29.1802902, -78.1159592, 78.0902634
19: -41.2959366, 19.3743992, -41.4630585, 19.5376720, -60.8336029, 60.8374519
20: -35.3733521, 21.7135620, -35.4727325, 21.8370667, -57.2104187, 57.1862907
21: -49.1210327, 25.3480797, -49.2869110, 25.4995995, -74.6206284, 74.6349945
22: -50.8869514, 29.8943977, -51.1011200, 30.1401958, -81.0271378, 80.9955139
23: -39.0517120, 26.4462891, -39.2523384, 26.6341648, -65.6858749, 65.6986237
24: -45.1124153, 22.7060871, -45.3349991, 22.8871269, -67.9995422, 68.0410843
25: -38.4321823, 30.8405571, -38.6250763, 31.0891647, -69.5213470, 69.4656372
26: -58.9803162, 37.3738060, -59.2143936, 37.7186699, -96.6989899, 96.5881958
27: -49.3207092, 27.2875214, -49.4755821, 27.3972549, -76.7179642, 76.7631073
28: -37.7799530, 28.6912651, -37.9485016, 28.8745728, -66.6545258, 66.6397705
29: -55.3307915, 34.2082863, -55.5817528, 34.4228783, -89.7536697, 89.7900391
30: -47.6956253, 27.1397152, -47.8900757, 27.2925377, -74.9881592, 75.0297928
31: -48.9116516, 23.9144173, -49.1294289, 24.0690937, -72.9807434, 73.0438461
32: -49.1010284, 27.4056606, -49.2205849, 27.5141773, -76.6152039, 76.6262360
33: -71.7458954, 43.9086533, -71.9342346, 44.1297188, -115.8756104, 115.8428879
34: -60.8241539, 29.8697052, -60.9999847, 30.1010246, -90.9251785, 90.8696899
35: -57.1245041, 34.5813217, -57.2883530, 34.7713242, -91.8958206, 91.8696747
36: -57.2345963, 33.7938232, -57.3670731, 34.0133171, -91.2479095, 91.1608963
37: -85.0800476, 32.8652992, -85.3775787, 33.1897354, -118.2697754, 118.2428741
38: -69.0547333, 40.8608551, -69.1906738, 41.0218544, -110.0765686, 110.0515289
39: -84.9636536, 40.6756134, -85.1354828, 40.8260918, -125.7897491, 125.8110886
40: -75.1473694, 29.9525547, -75.2901611, 30.0645828, -105.2119522, 105.2427063
41: -54.2319183, 25.8614845, -54.3916168, 26.0406837, -80.2725983, 80.2530975
42: -38.8940659, 29.3407173, -38.9909668, 29.4781551, -68.3722229, 68.3316803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 599
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 645
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 645
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 728

## Relational analysis of IS_A1_A2_B2_A1_B2_B1

### Relational analysis result of IS_A1_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.8597580, upper bound: 38.9270104
time: 76.01 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.8615592, upper bound: 38.9957076
time: 71.99 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 150.41 seconds
IS_A1_A1_A1_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 150.41
Output dim: 2, lower bound: -38.7265986, upper bound: 38.9172982
IS_A1_A1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 150.41
Output dim: 2, lower bound: -38.7265986, upper bound: 38.9857964
IS_A1_A1_A1_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 150.41
Output dim: 2, lower bound: -38.7597084, upper bound: 38.9188074
IS_A1_A1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 150.41
Output dim: 2, lower bound: -38.7265986, upper bound: 38.9873174
IS_A1_A1_A1_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 150.41
Output dim: 2, lower bound: -38.7991873, upper bound: 38.9218540
IS_A1_A1_A1_A2_B2_B2, status: Status.VERIFIED, split count: 6, time: 150.41
Output dim: 2, lower bound: -38.8007729, upper bound: 38.8809956
IS_A1_A2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 150.41
Output dim: 2, lower bound: -38.8597580, upper bound: 38.9270104
IS_A1_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 150.41
Output dim: 2, lower bound: -38.8615592, upper bound: 38.9957076

## BFS IS instance: IS_A1_A1_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -52.7293625, 42.7270432, -53.2778740, 43.0427933, -95.7721481, 96.0049057
1: -31.2208271, 35.8097801, -31.5851269, 36.0753021, -67.2961273, 67.3948975
2: -29.9012661, 35.2931175, -30.3241272, 35.6462021, -65.5474701, 65.6172485
3: -33.3567314, 41.1722031, -33.8336792, 41.6334038, -74.9901352, 75.0058823
4: -39.5141602, 38.5790939, -39.9578781, 38.9548035, -78.4689560, 78.5369720
5: -36.3203125, 40.9645538, -36.7853127, 41.4157486, -77.7360611, 77.7498627
6: -55.7027931, 22.1842842, -55.9548569, 22.4298534, -78.1326447, 78.1391373
7: -42.3767319, 39.8947716, -42.8440170, 40.2290268, -82.6057434, 82.7387848
8: -38.8191071, 45.1345139, -39.2881165, 45.5505714, -84.3696747, 84.4226303
9: -33.7467651, 37.3431015, -34.1295395, 37.5248642, -71.2716217, 71.4726410
10: -54.8596191, 51.9178543, -55.2111092, 52.3019714, -107.1615906, 107.1289673
11: -56.1083488, 39.2025452, -56.5491257, 39.6140900, -95.7224426, 95.7516708
12: -58.7671585, 43.4567566, -59.2022018, 43.9366760, -102.7038345, 102.6589508
13: -48.1464958, 49.1685524, -48.6394806, 49.6239090, -97.7704010, 97.8080292
14: -81.0018921, 42.9801941, -81.5156021, 43.3446617, -124.3465576, 124.4957962
15: -39.9933739, 36.1939392, -40.3435593, 36.3941193, -76.3874969, 76.5374908
16: -57.8436356, 40.7158813, -58.2544975, 40.8710556, -98.7146835, 98.9703827
17: -84.8622284, 62.1986694, -85.2315369, 62.5099792, -147.3722076, 147.4302063
18: -48.6656914, 28.5092545, -49.0324440, 29.0002918, -77.6659851, 77.5416946
19: -41.0776520, 19.1086693, -41.4234543, 19.4216957, -60.4993439, 60.5321198
20: -35.1828918, 21.4206161, -35.4331512, 21.7287292, -56.9116211, 56.8537636
21: -48.8649368, 25.0490952, -49.2327652, 25.3799152, -74.2448502, 74.2818604
22: -50.6512299, 29.5966549, -51.0321159, 29.9908409, -80.6420670, 80.6287613
23: -38.7736702, 26.1011600, -39.2118111, 26.4954147, -65.2690887, 65.3129730
24: -44.8306351, 22.4095516, -45.2794456, 22.7620506, -67.5926819, 67.6889954
25: -38.2046547, 30.5276222, -38.5707169, 30.9402676, -69.1449203, 69.0983429
26: -58.6519279, 36.7947998, -59.1513100, 37.4629097, -96.1148300, 95.9461060
27: -48.9883156, 26.8714790, -49.4216728, 27.2480049, -76.2363129, 76.2931519
28: -37.5652084, 28.3652668, -37.9092140, 28.7437973, -66.3089981, 66.2744827
29: -55.0599556, 33.8979683, -55.5119934, 34.2767792, -89.3367310, 89.4099579
30: -47.4457893, 26.8487377, -47.8284988, 27.1755733, -74.6213608, 74.6772308
31: -48.5848236, 23.6097450, -49.0706749, 23.9496021, -72.5344238, 72.6804199
32: -48.8356628, 27.1130505, -49.1768837, 27.3948498, -76.2305145, 76.2899323
33: -71.4923859, 43.7596016, -71.8709564, 44.0410614, -115.5334473, 115.6305542
34: -60.6221123, 29.5424900, -60.9464417, 29.9554253, -90.5775299, 90.4889297
35: -56.8963165, 34.3910599, -57.2219925, 34.6788635, -91.5751724, 91.6130524
36: -56.9798050, 33.5009041, -57.3070450, 33.8683929, -90.8481979, 90.8079529
37: -84.7517395, 32.6054497, -85.2963867, 33.0329285, -117.7846680, 117.9018326
38: -68.7318573, 40.4181519, -69.1157379, 40.8686066, -109.6004639, 109.5338898
39: -84.6003723, 40.4906693, -85.0333862, 40.7461319, -125.3465042, 125.5240555
40: -74.9162750, 29.7926083, -75.2245331, 29.9931049, -104.9093781, 105.0171432
41: -53.9819450, 25.5380669, -54.3482208, 25.9019833, -79.8839264, 79.8862839
42: -38.7298775, 29.0420761, -38.9616013, 29.3604584, -68.0903320, 68.0036774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=404, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 599
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1491
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 976

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 728

## Relational analysis of IS_A1_A1_A1_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_A1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.6808979, upper bound: 38.9843831
time: 77.80 seconds

## Relational analysis of IS_A1_A1_A1_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_A1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.6808979, upper bound: 38.9857967
time: 72.19 seconds

## BFS IS instance: IS_A1_A1_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -52.8274269, 42.7651443, -53.3021927, 43.0477371, -95.8751678, 96.0673370
1: -31.2744999, 35.8357811, -31.6006508, 36.0801888, -67.3546906, 67.4364319
2: -29.9978428, 35.3649445, -30.3540459, 35.6522446, -65.6500778, 65.7189941
3: -33.4751129, 41.2819061, -33.8717346, 41.6417046, -75.1168213, 75.1536407
4: -39.6142235, 38.6438255, -39.9881516, 38.9615097, -78.5757294, 78.6319733
5: -36.4384995, 41.0673141, -36.8232613, 41.4245682, -77.8630600, 77.8905792
6: -55.7638397, 22.2252350, -55.9685555, 22.4408760, -78.2047119, 78.1937866
7: -42.4619713, 39.9380493, -42.8708382, 40.2351074, -82.6970825, 82.8088837
8: -38.8842468, 45.1937637, -39.3063354, 45.5614662, -84.4457092, 84.5000992
9: -33.8206978, 37.3612900, -34.1499634, 37.5279083, -71.3486023, 71.5112457
10: -54.9607277, 52.0615768, -55.2254295, 52.3462181, -107.3069458, 107.2870026
11: -56.2131653, 39.3307838, -56.5573654, 39.6558838, -95.8690338, 95.8881531
12: -58.8567543, 43.5523262, -59.2125473, 43.9656296, -102.8223877, 102.7648697
13: -48.3513718, 49.3382607, -48.7052460, 49.6348114, -97.9861755, 98.0435028
14: -81.1322327, 43.0948715, -81.5338287, 43.3815536, -124.5137787, 124.6286926
15: -40.0517120, 36.2330894, -40.3599777, 36.4023399, -76.4540558, 76.5930634
16: -57.9196205, 40.7281265, -58.2738190, 40.8745232, -98.7941437, 99.0019455
17: -84.9218597, 62.2739410, -85.2462006, 62.5280991, -147.4499512, 147.5201263
18: -48.7694054, 28.6659584, -49.0443764, 29.0500622, -77.8194580, 77.7103348
19: -41.1589127, 19.1871376, -41.4319687, 19.4476757, -60.6065903, 60.6191063
20: -35.2407684, 21.5019913, -35.4408875, 21.7529926, -56.9937515, 56.9428787
21: -48.9554520, 25.1432648, -49.2430420, 25.4104805, -74.3659363, 74.3863068
22: -50.7371254, 29.6846123, -51.0431824, 30.0187035, -80.7558289, 80.7277908
23: -38.8767891, 26.2151928, -39.2192841, 26.5320435, -65.4088287, 65.4344788
24: -44.9426765, 22.5136375, -45.2885666, 22.7948818, -67.7375565, 67.8022003
25: -38.2861786, 30.6242065, -38.5798264, 30.9704170, -69.2565918, 69.2040253
26: -58.7600632, 36.9733124, -59.1605530, 37.5182915, -96.2783356, 96.1338577
27: -49.1237144, 27.0114517, -49.4336166, 27.2937393, -76.4174500, 76.4450684
28: -37.6500931, 28.4707794, -37.9162598, 28.7759323, -66.4260254, 66.3870316
29: -55.1686058, 34.0077133, -55.5230942, 34.3118668, -89.4804688, 89.5308075
30: -47.5281677, 26.9433537, -47.8358841, 27.2051830, -74.7333527, 74.7792358
31: -48.7095795, 23.6971493, -49.0840302, 23.9782104, -72.6877899, 72.7811737
32: -48.9134445, 27.1539383, -49.1898460, 27.4074612, -76.3209000, 76.3437805
33: -71.6268158, 43.8202286, -71.9121399, 44.0498466, -115.6766663, 115.7323685
34: -60.6772690, 29.6264896, -60.9583778, 29.9795208, -90.6567917, 90.5848694
35: -57.0063400, 34.4459991, -57.2549057, 34.6875763, -91.6939163, 91.7008972
36: -57.0607681, 33.5259323, -57.3281212, 33.8750992, -90.9358521, 90.8540497
37: -84.8424377, 32.6539116, -85.3162460, 33.0452118, -117.8876495, 117.9701538
38: -68.8140182, 40.4820862, -69.1373978, 40.8837204, -109.6977386, 109.6194839
39: -84.7467728, 40.5338478, -85.0715714, 40.7513390, -125.4980927, 125.6054153
40: -74.9870529, 29.8256664, -75.2399445, 30.0014458, -104.9884949, 105.0656128
41: -54.0544167, 25.6073036, -54.3602104, 25.9216957, -79.9761047, 79.9675064
42: -38.7848015, 29.1224213, -38.9688110, 29.3845406, -68.1693420, 68.0912323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=404, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1491
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 645
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 976

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 728

## Relational analysis of IS_A1_A1_A1_A1_A2_B2_A1

### Relational analysis result of IS_A1_A1_A1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.6808979, upper bound: 38.9858873
time: 82.38 seconds

## Relational analysis of IS_A1_A1_A1_A1_A2_B2_A2

### Relational analysis result of IS_A1_A1_A1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.7140511, upper bound: 38.9873176
time: 87.51 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -53.1347694, 42.9246597, -53.4551353, 43.0705719, -96.2053375, 96.3797913
1: -31.4730816, 35.9632149, -31.7123833, 36.1010437, -67.5741272, 67.6755981
2: -30.2343464, 35.5174789, -30.4799671, 35.6713524, -65.9057007, 65.9974442
3: -33.7421417, 41.4800377, -34.0099297, 41.6684189, -75.4105606, 75.4899673
4: -39.8787994, 38.8125992, -40.1163330, 38.9827003, -78.8614960, 78.9289322
5: -36.6941605, 41.2647705, -36.9559517, 41.4525146, -78.1466751, 78.2207031
6: -55.8873444, 22.4307690, -55.9983406, 22.5291119, -78.4164581, 78.4291077
7: -42.6986313, 40.0691299, -43.0167618, 40.2593269, -82.9579620, 83.0858917
8: -39.1406860, 45.3737488, -39.4634590, 45.5903702, -84.7310562, 84.8372040
9: -34.0669556, 37.4511070, -34.2511292, 37.5555801, -71.6225357, 71.7022400
10: -55.1145401, 52.2780495, -55.2895851, 52.4246635, -107.5391998, 107.5676346
11: -56.4067917, 39.6064224, -56.6037788, 39.7684555, -96.1752472, 96.2102051
12: -59.0677338, 43.8603287, -59.2413025, 44.1128693, -103.1806030, 103.1016312
13: -48.6485367, 49.5161705, -48.8135071, 49.6809387, -98.3294678, 98.3296814
14: -81.3225479, 43.2421455, -81.6602325, 43.4347534, -124.7573013, 124.9023743
15: -40.3053894, 36.3344688, -40.4763641, 36.4201508, -76.7255402, 76.8108368
16: -58.1554871, 40.8193207, -58.3567047, 40.9004784, -99.0559692, 99.1760254
17: -85.1126099, 62.3663559, -85.3342590, 62.5553169, -147.6679230, 147.7006073
18: -48.9351807, 28.9922581, -49.0953789, 29.1775627, -78.1127472, 78.0876389
19: -41.2956352, 19.3738995, -41.4618835, 19.5354595, -60.8310928, 60.8357849
20: -35.3729324, 21.7131691, -35.4711075, 21.8352947, -57.2082214, 57.1842766
21: -49.1206398, 25.3475933, -49.2853165, 25.4974785, -74.6181183, 74.6329117
22: -50.8859825, 29.8934479, -51.0971451, 30.1361294, -81.0221024, 80.9905853
23: -39.0514412, 26.4457130, -39.2511826, 26.6317081, -65.6831360, 65.6968994
24: -45.1118965, 22.7055759, -45.3327522, 22.8848324, -67.9967270, 68.0383301
25: -38.4314346, 30.8397274, -38.6219826, 31.0855293, -69.5169601, 69.4617081
26: -58.9795265, 37.3728409, -59.2112541, 37.7142181, -96.6937256, 96.5840912
27: -49.3202667, 27.2871590, -49.4738579, 27.3956585, -76.7159271, 76.7610168
28: -37.7796707, 28.6907063, -37.9473114, 28.8721733, -66.6518402, 66.6380157
29: -55.3299561, 34.2075882, -55.5783310, 34.4198303, -89.7497711, 89.7859116
30: -47.6950378, 27.1391563, -47.8877068, 27.2900772, -74.9851151, 75.0268631
31: -48.9112854, 23.9139595, -49.1278000, 24.0671673, -72.9784546, 73.0417633
32: -49.1005249, 27.4050636, -49.2185555, 27.5116425, -76.6121674, 76.6236115
33: -71.7452545, 43.9079094, -71.9314499, 44.1264420, -115.8716965, 115.8393555
34: -60.8236694, 29.8690529, -60.9978600, 30.0981445, -90.9218140, 90.8669128
35: -57.1238403, 34.5806885, -57.2855644, 34.7685471, -91.8923874, 91.8662491
36: -57.2339935, 33.7932587, -57.3645401, 34.0107536, -91.2447433, 91.1577988
37: -85.0792999, 32.8643608, -85.3744202, 33.1856079, -118.2649078, 118.2387848
38: -69.0541534, 40.8604584, -69.1881027, 41.0201988, -110.0743332, 110.0485611
39: -84.9629135, 40.6746979, -85.1322479, 40.8218307, -125.7847443, 125.8069458
40: -75.1468124, 29.9518623, -75.2879639, 30.0615196, -105.2083282, 105.2398224
41: -54.2314911, 25.8607655, -54.3899384, 26.0374374, -80.2689209, 80.2507019
42: -38.8936691, 29.3400459, -38.9894638, 29.4751434, -68.3688126, 68.3295059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=208, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 599
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1491
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 645
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 605

## Relational analysis of IS_A1_A2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.8197255, upper bound: 38.9925021
time: 80.67 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.8604784, upper bound: 38.9946249
time: 73.32 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 156.55 seconds
IS_A1_A1_A1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 156.55
Output dim: 2, lower bound: -38.6808979, upper bound: 38.9843831
IS_A1_A1_A1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 156.55
Output dim: 2, lower bound: -38.6808979, upper bound: 38.9857967
IS_A1_A1_A1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 156.55
Output dim: 2, lower bound: -38.6808979, upper bound: 38.9858873
IS_A1_A1_A1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 156.55
Output dim: 2, lower bound: -38.7140511, upper bound: 38.9873176
IS_A1_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 156.55
Output dim: 2, lower bound: -38.8197255, upper bound: 38.9925021
IS_A1_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 156.55
Output dim: 2, lower bound: -38.8604784, upper bound: 38.9946249

## BFS IS instance: IS_A1_A1_A1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -52.5900726, 42.6597939, -53.2778740, 43.0427933, -95.6328659, 95.9376678
1: -31.1190872, 35.7529564, -31.5851269, 36.0753021, -67.1943893, 67.3380814
2: -29.7886086, 35.2304420, -30.3241272, 35.6462021, -65.4348145, 65.5545654
3: -33.2338028, 41.1047897, -33.8336792, 41.6334038, -74.8672028, 74.9384689
4: -39.4125862, 38.5226059, -39.9578781, 38.9548035, -78.3673859, 78.4804840
5: -36.1990318, 40.8951111, -36.7853127, 41.4157486, -77.6147766, 77.6804199
6: -55.6714516, 22.1529808, -55.9548569, 22.4298534, -78.1013031, 78.1078339
7: -42.2339706, 39.8101349, -42.8440170, 40.2290268, -82.4629822, 82.6541519
8: -38.6578293, 45.0293427, -39.2881165, 45.5505714, -84.2084045, 84.3174591
9: -33.6914864, 37.3088455, -34.1295395, 37.5248642, -71.2163544, 71.4383850
10: -54.7967644, 51.8861160, -55.2111092, 52.3019714, -107.0987396, 107.0972214
11: -56.0379982, 39.1489449, -56.5491257, 39.6140900, -95.6520844, 95.6980743
12: -58.7062225, 43.3480835, -59.2022018, 43.9366760, -102.6428909, 102.5502853
13: -48.0947266, 49.1196365, -48.6394806, 49.6239090, -97.7186203, 97.7591171
14: -80.8440247, 42.9096451, -81.5156021, 43.3446617, -124.1886749, 124.4252472
15: -39.9285927, 36.1614685, -40.3435593, 36.3941193, -76.3227081, 76.5050201
16: -57.7849503, 40.7011948, -58.2544975, 40.8710556, -98.6559982, 98.9556885
17: -84.7664566, 62.1600075, -85.2315369, 62.5099792, -147.2764282, 147.3915405
18: -48.5864029, 28.4369087, -49.0324440, 29.0002918, -77.5866928, 77.4693527
19: -41.0217667, 19.0484810, -41.4234543, 19.4216957, -60.4434586, 60.4719315
20: -35.1449699, 21.3739166, -35.4331512, 21.7287292, -56.8736992, 56.8070679
21: -48.8058472, 24.9942989, -49.2327652, 25.3799152, -74.1857605, 74.2270660
22: -50.5670471, 29.4799728, -51.0321159, 29.9908409, -80.5578842, 80.5120773
23: -38.7093353, 26.0339947, -39.2118111, 26.4954147, -65.2047501, 65.2458038
24: -44.7423248, 22.3367271, -45.2794456, 22.7620506, -67.5043793, 67.6161652
25: -38.1199188, 30.4210052, -38.5707169, 30.9402676, -69.0601883, 68.9917221
26: -58.5466270, 36.6435280, -59.1513100, 37.4629097, -96.0095367, 95.7948380
27: -48.9433594, 26.8336697, -49.4216728, 27.2480049, -76.1913605, 76.2553406
28: -37.5022354, 28.2911301, -37.9092140, 28.7437973, -66.2460327, 66.2003479
29: -54.9716949, 33.8035545, -55.5119934, 34.2767792, -89.2484741, 89.3155441
30: -47.3578796, 26.7841129, -47.8284988, 27.1755733, -74.5334549, 74.6126099
31: -48.5092773, 23.5590668, -49.0706749, 23.9496021, -72.4588776, 72.6297455
32: -48.8032112, 27.0719986, -49.1768837, 27.3948498, -76.1980591, 76.2488785
33: -71.4138031, 43.6564331, -71.8709564, 44.0410614, -115.4548645, 115.5273895
34: -60.5503654, 29.4452438, -60.9464417, 29.9554253, -90.5057907, 90.3916855
35: -56.8176918, 34.3022690, -57.2219925, 34.6788635, -91.4965439, 91.5242615
36: -56.9223557, 33.4077225, -57.3070450, 33.8683929, -90.7907486, 90.7147675
37: -84.6441956, 32.4597664, -85.2963867, 33.0329285, -117.6771240, 117.7561493
38: -68.6822510, 40.3671417, -69.1157379, 40.8686066, -109.5508423, 109.4828720
39: -84.5406342, 40.4239273, -85.0333862, 40.7461319, -125.2867584, 125.4573059
40: -74.8689499, 29.7520466, -75.2245331, 29.9931049, -104.8620529, 104.9765778
41: -53.9357681, 25.4667587, -54.3482208, 25.9019833, -79.8377533, 79.8149796
42: -38.7062111, 28.9925995, -38.9616013, 29.3604584, -68.0666656, 67.9541931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=205, inp2_unstable=210, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=403, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 645
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 976

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 727

## Relational analysis of IS_A1_A1_A1_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_A1_A1_A1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.6776441, upper bound: 38.9132281
time: 78.67 seconds

## Relational analysis of IS_A1_A1_A1_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_A1_A1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.6776441, upper bound: 38.9832894
time: 72.49 seconds

## BFS IS instance: IS_A1_A1_A1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -52.7258797, 42.7252350, -53.2778740, 43.0427933, -95.7686691, 96.0031052
1: -31.2183857, 35.8084145, -31.5851269, 36.0753021, -67.2936859, 67.3935394
2: -29.8986645, 35.2914200, -30.3241272, 35.6462021, -65.5448685, 65.6155396
3: -33.3537445, 41.1703262, -33.8336792, 41.6334038, -74.9871445, 75.0040054
4: -39.5115509, 38.5778427, -39.9578781, 38.9548035, -78.4663544, 78.5357208
5: -36.3175354, 40.9624786, -36.7853127, 41.4157486, -77.7332840, 77.7477875
6: -55.7009888, 22.1810951, -55.9548569, 22.4298534, -78.1308441, 78.1359482
7: -42.3729744, 39.8914909, -42.8440170, 40.2290268, -82.6019897, 82.7354965
8: -38.8154526, 45.1324997, -39.2881165, 45.5505714, -84.3660278, 84.4206161
9: -33.7451057, 37.3419647, -34.1295395, 37.5248642, -71.2699738, 71.4714966
10: -54.8567047, 51.9162903, -55.2111092, 52.3019714, -107.1586761, 107.1273956
11: -56.1060791, 39.1995773, -56.5491257, 39.6140900, -95.7201691, 95.7487030
12: -58.7646942, 43.4543381, -59.2022018, 43.9366760, -102.7013702, 102.6565323
13: -48.1441307, 49.1671104, -48.6394806, 49.6239090, -97.7680359, 97.8065948
14: -80.9975586, 42.9791565, -81.5156021, 43.3446617, -124.3422089, 124.4947586
15: -39.9898376, 36.1854553, -40.3435593, 36.3941193, -76.3839569, 76.5290146
16: -57.8321495, 40.7134171, -58.2544975, 40.8710556, -98.7032013, 98.9679108
17: -84.8589783, 62.1959686, -85.2315369, 62.5099792, -147.3689270, 147.4274902
18: -48.6639862, 28.5070839, -49.0324440, 29.0002918, -77.6642761, 77.5395279
19: -41.0766487, 19.1069126, -41.4234543, 19.4216957, -60.4983444, 60.5303650
20: -35.1814728, 21.4191780, -35.4331512, 21.7287292, -56.9102020, 56.8523293
21: -48.8635979, 25.0473614, -49.2327652, 25.3799152, -74.2435074, 74.2801208
22: -50.6477623, 29.5933666, -51.0321159, 29.9908409, -80.6386032, 80.6254807
23: -38.7727089, 26.0990925, -39.2118111, 26.4954147, -65.2681122, 65.3109055
24: -44.8287354, 22.4077263, -45.2794456, 22.7620506, -67.5907898, 67.6871643
25: -38.2020073, 30.5246773, -38.5707169, 30.9402676, -69.1422729, 69.0953979
26: -58.6491394, 36.7912750, -59.1513100, 37.4629097, -96.1120377, 95.9425812
27: -48.9868088, 26.8702240, -49.4216728, 27.2480049, -76.2348099, 76.2919006
28: -37.5641785, 28.3632336, -37.9092140, 28.7437973, -66.3079758, 66.2724457
29: -55.0570297, 33.8955498, -55.5119934, 34.2767792, -89.3338089, 89.4075470
30: -47.4438095, 26.8467617, -47.8284988, 27.1755733, -74.6193771, 74.6752548
31: -48.5834732, 23.6081142, -49.0706749, 23.9496021, -72.5330734, 72.6787872
32: -48.8339462, 27.1109848, -49.1768837, 27.3948498, -76.2287903, 76.2878571
33: -71.4900970, 43.7569237, -71.8709564, 44.0410614, -115.5311584, 115.6278687
34: -60.6204071, 29.5401573, -60.9464417, 29.9554253, -90.5758362, 90.4866028
35: -56.8939743, 34.3888092, -57.2219925, 34.6788635, -91.5728378, 91.6107941
36: -56.9776421, 33.4988480, -57.3070450, 33.8683929, -90.8460312, 90.8058929
37: -84.7491302, 32.6021423, -85.2963867, 33.0329285, -117.7820587, 117.8985291
38: -68.7297821, 40.4167442, -69.1157379, 40.8686066, -109.5983734, 109.5324783
39: -84.5976868, 40.4872627, -85.0333862, 40.7461319, -125.3438187, 125.5206451
40: -74.9143753, 29.7901611, -75.2245331, 29.9931049, -104.9074707, 105.0146942
41: -53.9805641, 25.5354729, -54.3482208, 25.9019833, -79.8825455, 79.8836899
42: -38.7285995, 29.0396347, -38.9616013, 29.3604584, -68.0890579, 68.0012360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=205, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=404, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 599
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 976

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 727

## Relational analysis of IS_A1_A1_A1_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_A1_A1_A1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.6776441, upper bound: 38.8725740
time: 162.73 seconds

## Relational analysis of IS_A1_A1_A1_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_A1_A1_A1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.6776441, upper bound: 38.9177040
time: 72.67 seconds

## BFS IS instance: IS_A1_A1_A1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -52.6881371, 42.6978989, -53.3021927, 43.0477371, -95.7358704, 96.0000916
1: -31.1727448, 35.7789612, -31.6006508, 36.0801888, -67.2529297, 67.3796082
2: -29.8851814, 35.3023033, -30.3540459, 35.6522446, -65.5374298, 65.6563492
3: -33.3522110, 41.2145157, -33.8717346, 41.6417046, -74.9939117, 75.0862503
4: -39.5126534, 38.5873337, -39.9881516, 38.9615097, -78.4741669, 78.5754852
5: -36.3172684, 40.9978752, -36.8232613, 41.4245682, -77.7418289, 77.8211365
6: -55.7324409, 22.1939087, -55.9685555, 22.4408760, -78.1733170, 78.1624603
7: -42.3192062, 39.8534546, -42.8708382, 40.2351074, -82.5543137, 82.7242889
8: -38.7229576, 45.0886002, -39.3063354, 45.5614662, -84.2844238, 84.3949356
9: -33.7654114, 37.3270187, -34.1499634, 37.5279083, -71.2933197, 71.4769821
10: -54.8979645, 52.0297737, -55.2254295, 52.3462181, -107.2441864, 107.2551956
11: -56.1428032, 39.2771912, -56.5573654, 39.6558838, -95.7986908, 95.8345566
12: -58.7958679, 43.4436340, -59.2125473, 43.9656296, -102.7614899, 102.6561813
13: -48.2995796, 49.2893333, -48.7052460, 49.6348114, -97.9343872, 97.9945679
14: -80.9743576, 43.0242882, -81.5338287, 43.3815536, -124.3558960, 124.5581131
15: -39.9869308, 36.2006149, -40.3599777, 36.4023399, -76.3892670, 76.5605850
16: -57.8608894, 40.7134171, -58.2738190, 40.8745232, -98.7354126, 98.9872360
17: -84.8261108, 62.2352715, -85.2462006, 62.5280991, -147.3542023, 147.4814758
18: -48.6902046, 28.5935974, -49.0443764, 29.0500622, -77.7402649, 77.6379700
19: -41.1030273, 19.1269379, -41.4319687, 19.4476757, -60.5507050, 60.5589066
20: -35.2028694, 21.4552956, -35.4408875, 21.7529926, -56.9558640, 56.8961830
21: -48.8963699, 25.0884533, -49.2430420, 25.4104805, -74.3068542, 74.3314896
22: -50.6529770, 29.5679054, -51.0431824, 30.0187035, -80.6716766, 80.6110840
23: -38.8124619, 26.1480370, -39.2192841, 26.5320435, -65.3444977, 65.3673172
24: -44.8543663, 22.4408188, -45.2885666, 22.7948818, -67.6492462, 67.7293854
25: -38.2014771, 30.5175915, -38.5798264, 30.9704170, -69.1718903, 69.0974197
26: -58.6547661, 36.8219910, -59.1605530, 37.5182915, -96.1730576, 95.9825439
27: -49.0787888, 26.9736118, -49.4336166, 27.2937393, -76.3725281, 76.4072266
28: -37.5871506, 28.3966293, -37.9162598, 28.7759323, -66.3630829, 66.3128891
29: -55.0803604, 33.9132767, -55.5230942, 34.3118668, -89.3922272, 89.4363708
30: -47.4402695, 26.8787231, -47.8358841, 27.2051830, -74.6454468, 74.7146072
31: -48.6340904, 23.6464729, -49.0840302, 23.9782104, -72.6123047, 72.7304993
32: -48.8810463, 27.1129093, -49.1898460, 27.4074612, -76.2885056, 76.3027496
33: -71.5482025, 43.7171097, -71.9121399, 44.0498466, -115.5980377, 115.6292419
34: -60.6055450, 29.5292435, -60.9583778, 29.9795208, -90.5850525, 90.4876175
35: -56.9276657, 34.3572083, -57.2549057, 34.6875763, -91.6152344, 91.6121063
36: -57.0032768, 33.4327240, -57.3281212, 33.8750992, -90.8783722, 90.7608414
37: -84.7348785, 32.5082245, -85.3162460, 33.0452118, -117.7800903, 117.8244629
38: -68.7644119, 40.4311371, -69.1373978, 40.8837204, -109.6481323, 109.5685349
39: -84.6869659, 40.4671555, -85.0715714, 40.7513390, -125.4383087, 125.5387268
40: -74.9397888, 29.7851238, -75.2399445, 30.0014458, -104.9412384, 105.0250702
41: -54.0082321, 25.5359592, -54.3602104, 25.9216957, -79.9299164, 79.8961639
42: -38.7611580, 29.0729675, -38.9688110, 29.3845406, -68.1456985, 68.0417786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=205, inp2_unstable=210, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=403, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 599
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 645
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 727

## Relational analysis of IS_A1_A1_A1_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A1_A1_A1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.7106463, upper bound: 38.9147161
time: 78.17 seconds

## Relational analysis of IS_A1_A1_A1_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A1_A1_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.7121886, upper bound: 38.9847927
time: 74.70 seconds

## BFS IS instance: IS_A1_A1_A1_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -52.8239441, 42.7632942, -53.3021927, 43.0477371, -95.8716736, 96.0654907
1: -31.2720604, 35.8344040, -31.6006508, 36.0801888, -67.3522491, 67.4350586
2: -29.9952240, 35.3632584, -30.3540459, 35.6522446, -65.6474686, 65.7173004
3: -33.4721375, 41.2800026, -33.8717346, 41.6417046, -75.1138306, 75.1517334
4: -39.6116066, 38.6425819, -39.9881516, 38.9615097, -78.5731125, 78.6307220
5: -36.4357491, 41.0652237, -36.8232613, 41.4245682, -77.8602982, 77.8884888
6: -55.7620010, 22.2220535, -55.9685555, 22.4408760, -78.2028809, 78.1906052
7: -42.4582214, 39.9347534, -42.8708382, 40.2351074, -82.6933289, 82.8055878
8: -38.8805923, 45.1917725, -39.3063354, 45.5614662, -84.4420624, 84.4981079
9: -33.8190460, 37.3601456, -34.1499634, 37.5279083, -71.3469467, 71.5101089
10: -54.9577637, 52.0599861, -55.2254295, 52.3462181, -107.3039856, 107.2854156
11: -56.2108688, 39.3278198, -56.5573654, 39.6558838, -95.8667526, 95.8851852
12: -58.8543015, 43.5499115, -59.2125473, 43.9656296, -102.8199310, 102.7624588
13: -48.3490143, 49.3368149, -48.7052460, 49.6348114, -97.9838257, 98.0420532
14: -81.1278381, 43.0938110, -81.5338287, 43.3815536, -124.5093842, 124.6276245
15: -40.0481529, 36.2245865, -40.3599777, 36.4023399, -76.4504852, 76.5845642
16: -57.9081230, 40.7256508, -58.2738190, 40.8745232, -98.7826462, 98.9994659
17: -84.9186096, 62.2712555, -85.2462006, 62.5280991, -147.4467163, 147.5174561
18: -48.7677040, 28.6637821, -49.0443764, 29.0500622, -77.8177643, 77.7081604
19: -41.1579132, 19.1853809, -41.4319687, 19.4476757, -60.6055908, 60.6173477
20: -35.2393646, 21.5005455, -35.4408875, 21.7529926, -56.9923477, 56.9414291
21: -48.9540863, 25.1415215, -49.2430420, 25.4104805, -74.3645630, 74.3845520
22: -50.7336693, 29.6813202, -51.0431824, 30.0187035, -80.7523727, 80.7245026
23: -38.8758316, 26.2131157, -39.2192841, 26.5320435, -65.4078674, 65.4324036
24: -44.9407654, 22.5117950, -45.2885666, 22.7948818, -67.7356339, 67.8003540
25: -38.2835312, 30.6212730, -38.5798264, 30.9704170, -69.2539520, 69.2010956
26: -58.7572594, 36.9697647, -59.1605530, 37.5182915, -96.2755432, 96.1303177
27: -49.1222115, 27.0101738, -49.4336166, 27.2937393, -76.4159546, 76.4437866
28: -37.6490784, 28.4687386, -37.9162598, 28.7759323, -66.4250107, 66.3849945
29: -55.1656914, 34.0052986, -55.5230942, 34.3118668, -89.4775543, 89.5283966
30: -47.5262070, 26.9413757, -47.8358841, 27.2051830, -74.7313843, 74.7772598
31: -48.7082329, 23.6955204, -49.0840302, 23.9782104, -72.6864395, 72.7795410
32: -48.9117393, 27.1518669, -49.1898460, 27.4074612, -76.3191986, 76.3417053
33: -71.6245575, 43.8175621, -71.9121399, 44.0498466, -115.6744080, 115.7296982
34: -60.6755295, 29.6241398, -60.9583778, 29.9795208, -90.6550446, 90.5825195
35: -57.0039825, 34.4437332, -57.2549057, 34.6875763, -91.6915588, 91.6986389
36: -57.0586281, 33.5238914, -57.3281212, 33.8750992, -90.9337158, 90.8520050
37: -84.8397903, 32.6506119, -85.3162460, 33.0452118, -117.8850021, 117.9668579
38: -68.8119507, 40.4807129, -69.1373978, 40.8837204, -109.6956635, 109.6181107
39: -84.7440796, 40.5304871, -85.0715714, 40.7513390, -125.4954071, 125.6020508
40: -74.9851990, 29.8232327, -75.2399445, 30.0014458, -104.9866486, 105.0631790
41: -54.0530319, 25.6046982, -54.3602104, 25.9216957, -79.9747162, 79.9649048
42: -38.7835236, 29.1199932, -38.9688110, 29.3845406, -68.1680603, 68.0888062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=205, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=404, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1491
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 645
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 976

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 727

## Relational analysis of IS_A1_A1_A1_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A1_A1_A1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.7106463, upper bound: 38.8740703
time: 79.13 seconds

## Relational analysis of IS_A1_A1_A1_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A1_A1_A1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.7121886, upper bound: 38.9192277
time: 82.83 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -53.0266914, 42.8816795, -53.4280853, 43.0640030, -96.0906982, 96.3097687
1: -31.4133568, 35.9304504, -31.6951466, 36.0939217, -67.5072784, 67.6255951
2: -30.1311722, 35.4420166, -30.4480667, 35.6641502, -65.7953186, 65.8900833
3: -33.6152802, 41.3664169, -33.9692688, 41.6587906, -75.2740707, 75.3356781
4: -39.7719955, 38.7426453, -40.0839729, 38.9743500, -78.7463455, 78.8266144
5: -36.5670090, 41.1584320, -36.9153366, 41.4425392, -78.0095367, 78.0737686
6: -55.8118668, 22.3792610, -55.9800949, 22.5149136, -78.3267746, 78.3593597
7: -42.6045914, 40.0224380, -42.9872856, 40.2521591, -82.8567505, 83.0097198
8: -39.0661888, 45.3079605, -39.4423218, 45.5772972, -84.6434860, 84.7502823
9: -33.9857178, 37.4215965, -34.2284927, 37.5489006, -71.5346222, 71.6500854
10: -55.0093536, 52.1224709, -55.2736359, 52.3768234, -107.3861771, 107.3961029
11: -56.2922173, 39.4688148, -56.5926018, 39.7242203, -96.0164337, 96.0614166
12: -58.9724960, 43.7548065, -59.2291641, 44.0809402, -103.0534363, 102.9839706
13: -48.4292641, 49.3380661, -48.7431221, 49.6672707, -98.0965347, 98.0811768
14: -81.1807861, 43.1162720, -81.6390839, 43.3950348, -124.5758209, 124.7553406
15: -40.2360458, 36.2886810, -40.4564247, 36.4096603, -76.6457062, 76.7450943
16: -58.0692558, 40.7722473, -58.3342514, 40.8856430, -98.9548950, 99.1064987
17: -85.0408630, 62.2729721, -85.3163757, 62.5300522, -147.5708923, 147.5893402
18: -48.8226318, 28.8246422, -49.0804977, 29.1242867, -77.9469147, 77.9051361
19: -41.2088165, 19.2892609, -41.4516830, 19.5073509, -60.7161636, 60.7409439
20: -35.3088493, 21.6258640, -35.4615402, 21.8091259, -57.1179733, 57.0874023
21: -49.0219879, 25.2471867, -49.2726402, 25.4648743, -74.4868622, 74.5198288
22: -50.7926979, 29.7990093, -51.0839195, 30.1061783, -80.8988724, 80.8829269
23: -38.9424210, 26.3244114, -39.2418442, 26.5926914, -65.5351105, 65.5662537
24: -44.9924660, 22.5945282, -45.3212509, 22.8497849, -67.8422546, 67.9157791
25: -38.3438835, 30.7361889, -38.6110764, 31.0530720, -69.3969574, 69.3472672
26: -58.8628883, 37.1823654, -59.1991577, 37.6551437, -96.5180359, 96.3815231
27: -49.1756897, 27.1388874, -49.4588928, 27.3473587, -76.5230484, 76.5977783
28: -37.6880035, 28.5787735, -37.9381523, 28.8379860, -66.5259857, 66.5169220
29: -55.2126770, 34.0905533, -55.5646133, 34.3826065, -89.5952759, 89.6551590
30: -47.6000595, 27.0377464, -47.8763428, 27.2583008, -74.8583603, 74.9140854
31: -48.7799606, 23.8202705, -49.1124916, 24.0365887, -72.8165512, 72.9327621
32: -49.0167007, 27.3564644, -49.2037315, 27.4967728, -76.5134659, 76.5601959
33: -71.5857086, 43.8407936, -71.8830719, 44.1161308, -115.7018433, 115.7238617
34: -60.7606125, 29.7782784, -60.9833527, 30.0720100, -90.8326111, 90.7616272
35: -56.9940338, 34.5213394, -57.2466812, 34.7584724, -91.7525024, 91.7680206
36: -57.1377335, 33.7646637, -57.3375664, 34.0029984, -91.1407166, 91.1022339
37: -84.9799500, 32.8058624, -85.3516388, 33.1694870, -118.1494370, 118.1575012
38: -68.9531708, 40.7856216, -69.1599579, 41.0016022, -109.9547729, 109.9455795
39: -84.7938461, 40.6259155, -85.0866241, 40.8150558, -125.6089020, 125.7125397
40: -75.0675659, 29.9120064, -75.2697449, 30.0512199, -105.1187897, 105.1817474
41: -54.1527481, 25.7809944, -54.3758430, 26.0138092, -80.1665497, 80.1568375
42: -38.8326187, 29.2502632, -38.9806061, 29.4477501, -68.2803650, 68.2308655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=207, inp2_unstable=208, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=405, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 728

## Relational analysis of IS_A1_A2_B2_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_A2_B2_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.7723641, upper bound: 38.9909765
time: 74.94 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_A2_B2_A1_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.7723641, upper bound: 38.9238325
time: 71.75 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -53.1273422, 42.9203262, -53.4525299, 43.0691261, -96.1964569, 96.3728485
1: -31.4683189, 35.9570312, -31.7107296, 36.0989571, -67.5672760, 67.6677551
2: -30.2286720, 35.5144844, -30.4780521, 35.6703110, -65.8989868, 65.9925385
3: -33.7349014, 41.4764404, -34.0074463, 41.6671982, -75.4020844, 75.4838867
4: -39.8729286, 38.8083115, -40.1143188, 38.9812202, -78.8541489, 78.9226227
5: -36.6865768, 41.2616577, -36.9533310, 41.4514694, -78.1380310, 78.2149811
6: -55.8747215, 22.4219723, -55.9940224, 22.5260315, -78.4007568, 78.4159927
7: -42.6912537, 40.0662689, -43.0141945, 40.2583618, -82.9496155, 83.0804596
8: -39.1324158, 45.3683548, -39.4606094, 45.5885162, -84.7209320, 84.8289642
9: -34.0611038, 37.4406433, -34.2490463, 37.5520744, -71.6131744, 71.6896820
10: -55.1100693, 52.2682724, -55.2880249, 52.4213486, -107.5314178, 107.5562973
11: -56.3989029, 39.5995216, -56.6010666, 39.7661476, -96.1650467, 96.2005920
12: -59.0630875, 43.8520279, -59.2396965, 44.1100578, -103.1731262, 103.0917206
13: -48.6361122, 49.5086708, -48.8092575, 49.6783943, -98.3145065, 98.3179321
14: -81.3144379, 43.2350082, -81.6574936, 43.4322968, -124.7467346, 124.8925018
15: -40.2955399, 36.3283539, -40.4729958, 36.4180641, -76.7136078, 76.8013458
16: -58.1470795, 40.7853966, -58.3537712, 40.8892365, -99.0363007, 99.1391678
17: -85.1040192, 62.3506622, -85.3312531, 62.5496521, -147.6536560, 147.6819153
18: -48.9275055, 28.9830170, -49.0926933, 29.1744251, -78.1019287, 78.0757141
19: -41.2911987, 19.3683014, -41.4603577, 19.5333691, -60.8245659, 60.8286591
20: -35.3679810, 21.7080536, -35.4694366, 21.8335495, -57.2015305, 57.1774902
21: -49.1141586, 25.3417950, -49.2830811, 25.4954834, -74.6096420, 74.6248779
22: -50.8805428, 29.8874779, -51.0952492, 30.1341209, -81.0146637, 80.9827271
23: -39.0464668, 26.4390659, -39.2494736, 26.6294155, -65.6758804, 65.6885376
24: -45.1055565, 22.6995621, -45.3305779, 22.8827534, -67.9883118, 68.0301361
25: -38.4266357, 30.8334484, -38.6203575, 31.0833836, -69.5100174, 69.4537964
26: -58.9720459, 37.3623962, -59.2086678, 37.7106361, -96.6826782, 96.5710449
27: -49.3122711, 27.2798824, -49.4711151, 27.3932152, -76.7054901, 76.7509918
28: -37.7742233, 28.6849957, -37.9454422, 28.8702087, -66.6444244, 66.6304321
29: -55.3233719, 34.2014503, -55.5760422, 34.4177895, -89.7411575, 89.7774963
30: -47.6845856, 27.1332531, -47.8840904, 27.2880745, -74.9726562, 75.0173416
31: -48.9059753, 23.9084167, -49.1259995, 24.0653000, -72.9712753, 73.0344162
32: -49.0956535, 27.3987579, -49.2168846, 27.5094719, -76.6051178, 76.6156464
33: -71.7290039, 43.9036827, -71.9257507, 44.1250229, -115.8540192, 115.8294373
34: -60.8167648, 29.8635406, -60.9954681, 30.0962543, -90.9130020, 90.8590012
35: -57.1081238, 34.5768967, -57.2801666, 34.7672615, -91.8753815, 91.8570557
36: -57.2186241, 33.7903976, -57.3590965, 34.0097656, -91.2283783, 91.1494904
37: -85.0719376, 32.8552628, -85.3718948, 33.1818085, -118.2537460, 118.2271576
38: -69.0361481, 40.8512573, -69.1819382, 41.0170555, -110.0531845, 110.0331955
39: -84.9426804, 40.6704483, -85.1254883, 40.8204002, -125.7630768, 125.7959213
40: -75.1395721, 29.9462433, -75.2854691, 30.0596275, -105.1991959, 105.2317123
41: -54.2260399, 25.8515511, -54.3880157, 26.0336609, -80.2597046, 80.2395630
42: -38.8893433, 29.3320522, -38.9879532, 29.4723129, -68.3616486, 68.3200073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=207, inp2_unstable=208, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=407, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 599
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 645
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 728

## Relational analysis of IS_A1_A2_B2_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_A2_B2_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.8132560, upper bound: 38.9931010
time: 66.60 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_A2_B2_A1_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.8132560, upper bound: 38.9288664
time: 91.28 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 160.35 seconds
IS_A1_A1_A1_A1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 160.35
Output dim: 2, lower bound: -38.6776441, upper bound: 38.9132281
IS_A1_A1_A1_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 160.35
Output dim: 2, lower bound: -38.6776441, upper bound: 38.9832894
IS_A1_A1_A1_A1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 160.35
Output dim: 2, lower bound: -38.6776441, upper bound: 38.8725740
IS_A1_A1_A1_A1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 160.35
Output dim: 2, lower bound: -38.6776441, upper bound: 38.9177040
IS_A1_A1_A1_A1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 160.35
Output dim: 2, lower bound: -38.7106463, upper bound: 38.9147161
IS_A1_A1_A1_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 160.35
Output dim: 2, lower bound: -38.7121886, upper bound: 38.9847927
IS_A1_A1_A1_A1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 160.35
Output dim: 2, lower bound: -38.7106463, upper bound: 38.8740703
IS_A1_A1_A1_A1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 160.35
Output dim: 2, lower bound: -38.7121886, upper bound: 38.9192277
IS_A1_A2_B2_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 160.35
Output dim: 2, lower bound: -38.7723641, upper bound: 38.9909765
IS_A1_A2_B2_A1_B2_B2_A1_A2, status: Status.VERIFIED, split count: 8, time: 160.35
Output dim: 2, lower bound: -38.7723641, upper bound: 38.9238325
IS_A1_A2_B2_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 160.35
Output dim: 2, lower bound: -38.8132560, upper bound: 38.9931010
IS_A1_A2_B2_A1_B2_B2_A2_A2, status: Status.VERIFIED, split count: 8, time: 160.35
Output dim: 2, lower bound: -38.8132560, upper bound: 38.9288664

## BFS IS instance: IS_A1_A1_A1_A1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -52.5873985, 42.6590195, -53.2685738, 43.0400238, -95.6274261, 95.9275970
1: -31.1173592, 35.7523880, -31.5789700, 36.0733871, -67.1907425, 67.3313599
2: -29.7870445, 35.2297821, -30.3186092, 35.6439133, -65.4309540, 65.5483932
3: -33.2317619, 41.1037827, -33.8265076, 41.6299706, -74.8617325, 74.9302826
4: -39.4106827, 38.5219498, -39.9514847, 38.9526215, -78.3632889, 78.4734344
5: -36.1972504, 40.8939133, -36.7789154, 41.4116364, -77.6088791, 77.6728287
6: -55.6704941, 22.1514702, -55.9515228, 22.4241123, -78.0946045, 78.1029968
7: -42.2315674, 39.8086395, -42.8355255, 40.2237473, -82.4553146, 82.6441650
8: -38.6558075, 45.0282745, -39.2809753, 45.5468750, -84.2026825, 84.3092499
9: -33.6901550, 37.3060913, -34.1249123, 37.5144386, -71.2045898, 71.4309998
10: -54.7945023, 51.8848076, -55.2027397, 52.2972603, -107.0917587, 107.0875473
11: -56.0349159, 39.1474991, -56.5374298, 39.6088257, -95.6437378, 95.6849289
12: -58.7047882, 43.3459320, -59.1971741, 43.9291763, -102.6339645, 102.5431061
13: -48.0931244, 49.1140404, -48.6337357, 49.6029396, -97.6960602, 97.7477722
14: -80.8411865, 42.9088211, -81.5054855, 43.3417511, -124.1829376, 124.4142914
15: -39.9266129, 36.1600037, -40.3365250, 36.3883553, -76.3149643, 76.4965134
16: -57.7787437, 40.6997757, -58.2324524, 40.8659668, -98.6447144, 98.9322281
17: -84.7620316, 62.1586151, -85.2160034, 62.5052032, -147.2672424, 147.3746033
18: -48.5852661, 28.4355354, -49.0285378, 28.9954910, -77.5807571, 77.4640732
19: -41.0208740, 19.0476456, -41.4202881, 19.4186325, -60.4395065, 60.4679337
20: -35.1412239, 21.3729019, -35.4192963, 21.7251720, -56.8663940, 56.7921982
21: -48.8029213, 24.9933128, -49.2218323, 25.3763752, -74.1792908, 74.2151337
22: -50.5648041, 29.4781170, -51.0241013, 29.9840126, -80.5488129, 80.5022125
23: -38.7085800, 26.0330009, -39.2091980, 26.4918346, -65.2004089, 65.2422028
24: -44.7410889, 22.3357353, -45.2751465, 22.7585754, -67.4996643, 67.6108856
25: -38.1178589, 30.4197731, -38.5636406, 30.9360847, -69.0539398, 68.9834137
26: -58.5449905, 36.6414871, -59.1455307, 37.4554977, -96.0004883, 95.7870178
27: -48.9380608, 26.8327751, -49.4031487, 27.2447548, -76.1828079, 76.2359238
28: -37.5013504, 28.2902756, -37.9060669, 28.7408772, -66.2422256, 66.1963425
29: -54.9684753, 33.8023300, -55.5003014, 34.2723389, -89.2408142, 89.3026276
30: -47.3567200, 26.7830143, -47.8243828, 27.1716633, -74.5283813, 74.6073990
31: -48.5065269, 23.5579700, -49.0603142, 23.9456272, -72.4521484, 72.6182785
32: -48.8019943, 27.0708199, -49.1726799, 27.3904438, -76.1924362, 76.2434998
33: -71.4122925, 43.6547813, -71.8656769, 44.0348930, -115.4471741, 115.5204620
34: -60.5488586, 29.4435730, -60.9410362, 29.9493504, -90.4982071, 90.3846054
35: -56.8160591, 34.3007202, -57.2161331, 34.6732788, -91.4893341, 91.5168533
36: -56.9208527, 33.4062080, -57.3017159, 33.8629990, -90.7838440, 90.7079239
37: -84.6422424, 32.4573555, -85.2896576, 33.0243912, -117.6666260, 117.7470093
38: -68.6808090, 40.3646431, -69.1105042, 40.8608093, -109.5416107, 109.4751358
39: -84.5388489, 40.4220581, -85.0271149, 40.7392235, -125.2780762, 125.4491730
40: -74.8674774, 29.7502003, -75.2194672, 29.9860611, -104.8535309, 104.9696579
41: -53.9348564, 25.4652157, -54.3450356, 25.8963242, -79.8311768, 79.8102417
42: -38.7053757, 28.9914246, -38.9586754, 29.3561325, -68.0615082, 67.9500961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=205, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=403, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1491
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 976

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 727

## Relational analysis of IS_A1_A1_A1_A1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_A1_A1_A1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.6244522, upper bound: 38.9814786
time: 62.09 seconds

## Relational analysis of IS_A1_A1_A1_A1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_A1_A1_A1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.6244522, upper bound: 38.9832896
time: 72.08 seconds

## BFS IS instance: IS_A1_A1_A1_A1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -52.6854515, 42.6971092, -53.2928772, 43.0449829, -95.7304306, 95.9899826
1: -31.1709938, 35.7783852, -31.5944729, 36.0782585, -67.2492523, 67.3728561
2: -29.8836174, 35.3016510, -30.3485222, 35.6499596, -65.5335770, 65.6501770
3: -33.3501740, 41.2134857, -33.8645592, 41.6382523, -74.9884262, 75.0780487
4: -39.5107422, 38.5866928, -39.9817619, 38.9593201, -78.4700623, 78.5684509
5: -36.3154526, 40.9966812, -36.8168564, 41.4204330, -77.7358856, 77.8135376
6: -55.7314720, 22.1924229, -55.9652519, 22.4351444, -78.1666107, 78.1576767
7: -42.3167992, 39.8519211, -42.8623505, 40.2298050, -82.5466003, 82.7142715
8: -38.7209244, 45.0875473, -39.2992325, 45.5577736, -84.2786865, 84.3867798
9: -33.7640762, 37.3242607, -34.1453362, 37.5174751, -71.2815552, 71.4695969
10: -54.8956528, 52.0285110, -55.2170181, 52.3415298, -107.2371826, 107.2455292
11: -56.1397209, 39.2757339, -56.5456619, 39.6506081, -95.7903137, 95.8213959
12: -58.7944031, 43.4415054, -59.2075233, 43.9581299, -102.7525330, 102.6490326
13: -48.2980080, 49.2837219, -48.6995087, 49.6138115, -97.9118195, 97.9832306
14: -80.9714966, 43.0234833, -81.5237122, 43.3786774, -124.3501740, 124.5471878
15: -39.9849701, 36.1991463, -40.3529320, 36.3965607, -76.3815308, 76.5520782
16: -57.8546562, 40.7119865, -58.2517624, 40.8694229, -98.7240753, 98.9637451
17: -84.8216858, 62.2338333, -85.2306442, 62.5233231, -147.3450012, 147.4644775
18: -48.6890373, 28.5922241, -49.0404663, 29.0452538, -77.7342911, 77.6326904
19: -41.1021423, 19.1261139, -41.4288025, 19.4446220, -60.5467606, 60.5549164
20: -35.1991234, 21.4542694, -35.4270172, 21.7494507, -56.9485664, 56.8812866
21: -48.8934326, 25.0874634, -49.2320862, 25.4069519, -74.3003845, 74.3195496
22: -50.6507339, 29.5660515, -51.0351677, 30.0118847, -80.6626205, 80.6012115
23: -38.8117256, 26.1470222, -39.2166977, 26.5284691, -65.3401947, 65.3637161
24: -44.8531189, 22.4398308, -45.2842636, 22.7914162, -67.6445312, 67.7240906
25: -38.1994171, 30.5163670, -38.5727463, 30.9662495, -69.1656647, 69.0891113
26: -58.6531258, 36.8199463, -59.1547737, 37.5108681, -96.1639862, 95.9747162
27: -49.0734940, 26.9727154, -49.4150772, 27.2904682, -76.3639603, 76.3877945
28: -37.5862656, 28.3957939, -37.9131012, 28.7730236, -66.3592911, 66.3088989
29: -55.0771294, 33.9120560, -55.5114136, 34.3074188, -89.3845520, 89.4234695
30: -47.4390869, 26.8776073, -47.8318062, 27.2012749, -74.6403580, 74.7094116
31: -48.6313324, 23.6453781, -49.0736656, 23.9742508, -72.6055756, 72.7190399
32: -48.8798065, 27.1117363, -49.1856194, 27.4030380, -76.2828445, 76.2973480
33: -71.5467148, 43.7154465, -71.9068069, 44.0436668, -115.5903778, 115.6222534
34: -60.6040268, 29.5275364, -60.9529343, 29.9734917, -90.5775146, 90.4804688
35: -56.9260368, 34.3556404, -57.2490540, 34.6820602, -91.6080933, 91.6046906
36: -57.0018082, 33.4312134, -57.3228073, 33.8697701, -90.8715820, 90.7540207
37: -84.7329559, 32.5058365, -85.3095322, 33.0366287, -117.7695847, 117.8153687
38: -68.7629700, 40.4286423, -69.1321411, 40.8759308, -109.6388931, 109.5607834
39: -84.6851959, 40.4652748, -85.0653000, 40.7444916, -125.4296875, 125.5305786
40: -74.9383011, 29.7832451, -75.2348328, 29.9944000, -104.9327011, 105.0180817
41: -54.0073433, 25.5343933, -54.3570023, 25.9160576, -79.9234009, 79.8913956
42: -38.7603302, 29.0717773, -38.9658890, 29.3801994, -68.1405334, 68.0376663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=205, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=403, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 599
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 645
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 976

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 727

## Relational analysis of IS_A1_A1_A1_A1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_A1_A1_A1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.6574702, upper bound: 38.9829795
time: 89.70 seconds

## Relational analysis of IS_A1_A1_A1_A1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_A1_A1_A1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.6574702, upper bound: 38.9847929
time: 77.46 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -52.8873558, 42.8144341, -53.4280853, 43.0640030, -95.9513550, 96.2425156
1: -31.3116169, 35.8736610, -31.6951466, 36.0939217, -67.4055328, 67.5688095
2: -30.0184841, 35.3793983, -30.4480667, 35.6641502, -65.6826324, 65.8274689
3: -33.4923630, 41.2990608, -33.9692688, 41.6587906, -75.1511459, 75.2683258
4: -39.6704063, 38.6861496, -40.0839729, 38.9743500, -78.6447525, 78.7701263
5: -36.4457626, 41.0890388, -36.9153366, 41.4425392, -77.8883057, 78.0043793
6: -55.7805138, 22.3479347, -55.9800949, 22.5149136, -78.2954178, 78.3280334
7: -42.4618530, 39.9378471, -42.9872856, 40.2521591, -82.7140121, 82.9251328
8: -38.9049072, 45.2028084, -39.4423218, 45.5772972, -84.4822083, 84.6451263
9: -33.9303894, 37.3873444, -34.2284927, 37.5489006, -71.4792786, 71.6158371
10: -54.9463501, 52.0906830, -55.2736359, 52.3768234, -107.3231735, 107.3643188
11: -56.2218170, 39.4151611, -56.5926018, 39.7242203, -95.9460373, 96.0077667
12: -58.9116364, 43.6460876, -59.2291641, 44.0809402, -102.9925766, 102.8752518
13: -48.3774071, 49.2891388, -48.7431221, 49.6672707, -98.0446777, 98.0322571
14: -81.0226746, 43.0456886, -81.6390839, 43.3950348, -124.4177094, 124.6847687
15: -40.1711845, 36.2562103, -40.4564247, 36.4096603, -76.5808411, 76.7126312
16: -58.0104866, 40.7573853, -58.3342514, 40.8856430, -98.8961334, 99.0916367
17: -84.9449921, 62.2342987, -85.3163757, 62.5300522, -147.4750366, 147.5506744
18: -48.7433281, 28.7522392, -49.0804977, 29.1242867, -77.8676147, 77.8327332
19: -41.1529732, 19.2290726, -41.4516830, 19.5073509, -60.6603165, 60.6807480
20: -35.2709618, 21.5791397, -35.4615402, 21.8091259, -57.0800819, 57.0406799
21: -48.9629059, 25.1923637, -49.2726402, 25.4648743, -74.4277802, 74.4650040
22: -50.7086220, 29.6823502, -51.0839195, 30.1061783, -80.8147964, 80.7662659
23: -38.8780975, 26.2572651, -39.2418442, 26.5926914, -65.4707870, 65.4990997
24: -44.9041786, 22.5217113, -45.3212509, 22.8497849, -67.7539673, 67.8429642
25: -38.2592087, 30.6295967, -38.6110764, 31.0530720, -69.3122787, 69.2406769
26: -58.7577209, 37.0310593, -59.1991577, 37.6551437, -96.4128647, 96.2302170
27: -49.1307755, 27.1009998, -49.4588928, 27.3473587, -76.4781342, 76.5598907
28: -37.6250572, 28.5046272, -37.9381523, 28.8379860, -66.4630432, 66.4427795
29: -55.1244621, 33.9960823, -55.5646133, 34.3826065, -89.5070648, 89.5606918
30: -47.5121384, 26.9731140, -47.8763428, 27.2583008, -74.7704391, 74.8494415
31: -48.7045326, 23.7695084, -49.1124916, 24.0365887, -72.7411194, 72.8820038
32: -48.9842987, 27.3154736, -49.2037315, 27.4967728, -76.4810562, 76.5192032
33: -71.5071487, 43.7375832, -71.8830719, 44.1161308, -115.6232758, 115.6206512
34: -60.6889305, 29.6810398, -60.9833527, 30.0720100, -90.7609406, 90.6643906
35: -56.9153900, 34.4324646, -57.2466812, 34.7584724, -91.6738586, 91.6791458
36: -57.0803413, 33.6715164, -57.3375664, 34.0029984, -91.0833435, 91.0090790
37: -84.8724670, 32.6601868, -85.3516388, 33.1694870, -118.0419540, 118.0118256
38: -68.9037323, 40.7346725, -69.1599579, 41.0016022, -109.9053345, 109.8946304
39: -84.7341461, 40.5592117, -85.0866241, 40.8150558, -125.5492020, 125.6458359
40: -75.0202332, 29.8714085, -75.2697449, 30.0512199, -105.0714417, 105.1411514
41: -54.1066093, 25.7096519, -54.3758430, 26.0138092, -80.1204224, 80.0854950
42: -38.8089828, 29.2007828, -38.9806061, 29.4477501, -68.2567291, 68.1813889

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=208, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=404, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 645
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 976

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 727

## Relational analysis of IS_A1_A2_B2_A1_B2_B2_A1_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B2_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.7688733, upper bound: 38.9197571
time: 70.82 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2_B2_A1_A1_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.7704842, upper bound: 38.9898837
time: 77.50 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -52.9879951, 42.8530960, -53.4525299, 43.0691261, -96.0570984, 96.3056183
1: -31.3665829, 35.9002762, -31.7107296, 36.0989571, -67.4655380, 67.6110077
2: -30.1159992, 35.4518814, -30.4780521, 35.6703110, -65.7863083, 65.9299316
3: -33.6120110, 41.4091110, -34.0074463, 41.6671982, -75.2792053, 75.4165573
4: -39.7713470, 38.7518311, -40.1143188, 38.9812202, -78.7525558, 78.8661423
5: -36.5653419, 41.1922607, -36.9533310, 41.4514694, -78.0168152, 78.1455841
6: -55.8433456, 22.3906250, -55.9940224, 22.5260315, -78.3693771, 78.3846436
7: -42.5485153, 39.9817200, -43.0141945, 40.2583618, -82.8068771, 82.9959106
8: -38.9711304, 45.2632141, -39.4606094, 45.5885162, -84.5596466, 84.7238159
9: -34.0057526, 37.4063721, -34.2490463, 37.5520744, -71.5578308, 71.6554184
10: -55.0470810, 52.2364998, -55.2880249, 52.4213486, -107.4684219, 107.5245209
11: -56.3285027, 39.5458336, -56.6010666, 39.7661476, -96.0946503, 96.1468964
12: -59.0022621, 43.7433205, -59.2396965, 44.1100578, -103.1123047, 102.9830170
13: -48.5842438, 49.4597969, -48.8092575, 49.6783943, -98.2626343, 98.2690582
14: -81.1563721, 43.1643791, -81.6574936, 43.4322968, -124.5886688, 124.8218689
15: -40.2307205, 36.2958832, -40.4729958, 36.4180641, -76.6487808, 76.7688751
16: -58.0882950, 40.7705345, -58.3537712, 40.8892365, -98.9775085, 99.1243057
17: -85.0081711, 62.3119545, -85.3312531, 62.5496521, -147.5578308, 147.6432037
18: -48.8482361, 28.9105949, -49.0926933, 29.1744251, -78.0226593, 78.0032883
19: -41.2353363, 19.3081112, -41.4603577, 19.5333691, -60.7687035, 60.7684631
20: -35.3301163, 21.6613121, -35.4694366, 21.8335495, -57.1636658, 57.1307487
21: -49.0550957, 25.2869682, -49.2830811, 25.4954834, -74.5505829, 74.5700531
22: -50.7964745, 29.7708015, -51.0952492, 30.1341209, -80.9305954, 80.8660507
23: -38.9821548, 26.3718815, -39.2494736, 26.6294155, -65.6115723, 65.6213455
24: -45.0172806, 22.6267166, -45.3305779, 22.8827534, -67.9000320, 67.9572906
25: -38.3419647, 30.7268562, -38.6203575, 31.0833836, -69.4253464, 69.3471985
26: -58.8668938, 37.2110939, -59.2086678, 37.7106361, -96.5775223, 96.4197540
27: -49.2673836, 27.2419701, -49.4711151, 27.3932152, -76.6605911, 76.7130814
28: -37.7112999, 28.6108475, -37.9454422, 28.8702087, -66.5815048, 66.5562897
29: -55.2351723, 34.1069870, -55.5760422, 34.4177895, -89.6529617, 89.6830292
30: -47.5966682, 27.0686169, -47.8840904, 27.2880745, -74.8847427, 74.9527054
31: -48.8305855, 23.8576412, -49.1259995, 24.0653000, -72.8958893, 72.9836426
32: -49.0632896, 27.3577824, -49.2168846, 27.5094719, -76.5727539, 76.5746613
33: -71.6503983, 43.8005333, -71.9257507, 44.1250229, -115.7754211, 115.7262878
34: -60.7451134, 29.7662754, -60.9954681, 30.0962543, -90.8413696, 90.7617416
35: -57.0294266, 34.4880447, -57.2801666, 34.7672615, -91.7966766, 91.7682114
36: -57.1612320, 33.6972198, -57.3590965, 34.0097656, -91.1709976, 91.0563202
37: -84.9644089, 32.7095337, -85.3718948, 33.1818085, -118.1462173, 118.0814285
38: -68.9867401, 40.8003159, -69.1819382, 41.0170555, -110.0037918, 109.9822464
39: -84.8829346, 40.6037140, -85.1254883, 40.8204002, -125.7033386, 125.7291946
40: -75.0922394, 29.9056206, -75.2854691, 30.0596275, -105.1518478, 105.1910858
41: -54.1798706, 25.7801895, -54.3880157, 26.0336609, -80.2135315, 80.1682053
42: -38.8657227, 29.2825851, -38.9879532, 29.4723129, -68.3380280, 68.2705307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=208, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=406, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 599
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 727

## Relational analysis of IS_A1_A2_B2_A1_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.8096668, upper bound: 38.9218273
time: 75.68 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.7071205, upper bound: 38.9920078
time: 89.63 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 167.79 seconds
IS_A1_A1_A1_A1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 167.79
Output dim: 2, lower bound: -38.6244522, upper bound: 38.9814786
IS_A1_A1_A1_A1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 167.79
Output dim: 2, lower bound: -38.6244522, upper bound: 38.9832896
IS_A1_A1_A1_A1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 167.79
Output dim: 2, lower bound: -38.6574702, upper bound: 38.9829795
IS_A1_A1_A1_A1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 167.79
Output dim: 2, lower bound: -38.6574702, upper bound: 38.9847929
IS_A1_A2_B2_A1_B2_B2_A1_A1_B1, status: Status.VERIFIED, split count: 9, time: 167.79
Output dim: 2, lower bound: -38.7688733, upper bound: 38.9197571
IS_A1_A2_B2_A1_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 167.79
Output dim: 2, lower bound: -38.7704842, upper bound: 38.9898837
IS_A1_A2_B2_A1_B2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 9, time: 167.79
Output dim: 2, lower bound: -38.8096668, upper bound: 38.9218273
IS_A1_A2_B2_A1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 167.79
Output dim: 2, lower bound: -38.7071205, upper bound: 38.9920078

## BFS IS instance: IS_A1_A1_A1_A1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -52.3913116, 42.5930138, -53.2685738, 43.0400238, -95.4313354, 95.8615875
1: -30.9904270, 35.6907730, -31.5789700, 36.0733871, -67.0638046, 67.2697449
2: -29.6547623, 35.1667633, -30.3186092, 35.6439133, -65.2986755, 65.4853668
3: -33.0648193, 41.0119476, -33.8265076, 41.6299706, -74.6947937, 74.8384552
4: -39.2979088, 38.4649200, -39.9514847, 38.9526215, -78.2505264, 78.4164047
5: -36.0333786, 40.8041496, -36.7789154, 41.4116364, -77.4449997, 77.5830688
6: -55.6229134, 22.0761261, -55.9515228, 22.4241123, -78.0470276, 78.0276489
7: -42.0438004, 39.7045059, -42.8355255, 40.2237473, -82.2675400, 82.5400314
8: -38.4809227, 44.9039955, -39.2809753, 45.5468750, -84.0277863, 84.1849670
9: -33.6260681, 37.2584534, -34.1249123, 37.5144386, -71.1405029, 71.3833618
10: -54.7173576, 51.8278885, -55.2027397, 52.2972603, -107.0146179, 107.0306244
11: -55.9620743, 39.0943718, -56.5374298, 39.6088257, -95.5709000, 95.6318054
12: -58.6085739, 43.1708794, -59.1971741, 43.9291763, -102.5377502, 102.3680496
13: -48.0359573, 49.0544548, -48.6337357, 49.6029396, -97.6388779, 97.6881790
14: -80.6101074, 42.8065567, -81.5054855, 43.3417511, -123.9518585, 124.3120270
15: -39.8156013, 36.1141586, -40.3365250, 36.3883553, -76.2039566, 76.4506760
16: -57.6931839, 40.6414490, -58.2324524, 40.8659668, -98.5591507, 98.8739014
17: -84.6038361, 62.0936699, -85.2160034, 62.5052032, -147.1090393, 147.3096619
18: -48.4921684, 28.3484688, -49.0285378, 28.9954910, -77.4876556, 77.3769989
19: -40.9603081, 19.0051613, -41.4202881, 19.4186325, -60.3789368, 60.4254456
20: -35.0904160, 21.3332729, -35.4192963, 21.7251720, -56.8155708, 56.7525711
21: -48.7436600, 24.9657555, -49.2218323, 25.3763752, -74.1200333, 74.1875763
22: -50.4677353, 29.4017105, -51.0241013, 29.9840126, -80.4517441, 80.4258118
23: -38.6552811, 25.9851856, -39.2091980, 26.4918346, -65.1471100, 65.1943817
24: -44.6563301, 22.2817898, -45.2751465, 22.7585754, -67.4149017, 67.5569305
25: -38.0566025, 30.3461399, -38.5636406, 30.9360847, -68.9926910, 68.9097748
26: -58.4337158, 36.4624252, -59.1455307, 37.4554977, -95.8892059, 95.6079559
27: -48.8603973, 26.8012676, -49.4031487, 27.2447548, -76.1051483, 76.2044144
28: -37.4510269, 28.2372284, -37.9060669, 28.7408772, -66.1918945, 66.1432953
29: -54.8459511, 33.7487564, -55.5003014, 34.2723389, -89.1182861, 89.2490540
30: -47.2873230, 26.7278023, -47.8243828, 27.1716633, -74.4589844, 74.5521851
31: -48.4326477, 23.5391521, -49.0603142, 23.9456272, -72.3782730, 72.5994568
32: -48.7455559, 26.9918785, -49.1726799, 27.3904438, -76.1360016, 76.1645508
33: -71.3123474, 43.5110512, -71.8656769, 44.0348930, -115.3472443, 115.3767242
34: -60.4619598, 29.3147621, -60.9410362, 29.9493504, -90.4113083, 90.2557983
35: -56.7172546, 34.1757812, -57.2161331, 34.6732788, -91.3905182, 91.3919144
36: -56.8348007, 33.2713051, -57.3017159, 33.8629990, -90.6977997, 90.5730133
37: -84.4683380, 32.2193260, -85.2896576, 33.0243912, -117.4927292, 117.5089874
38: -68.6054993, 40.2563858, -69.1105042, 40.8608093, -109.4663086, 109.3668823
39: -84.4263000, 40.2858124, -85.0271149, 40.7392235, -125.1655273, 125.3129272
40: -74.7684708, 29.6256981, -75.2194672, 29.9860611, -104.7545319, 104.8451614
41: -53.8562813, 25.3344193, -54.3450356, 25.8963242, -79.7526093, 79.6794510
42: -38.6815834, 28.9258175, -38.9586754, 29.3561325, -68.0377197, 67.8844910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=204, inp2_unstable=209, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=401, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1491
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 1282

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 713

## Relational analysis of IS_A1_A1_A1_A1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_A1_A1_A1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.6202687, upper bound: 38.9282816
time: 106.78 seconds

## Relational analysis of IS_A1_A1_A1_A1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_A1_A1_A1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.6202687, upper bound: 38.9800464
time: 104.37 seconds

## BFS IS instance: IS_A1_A1_A1_A1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -52.5807838, 42.6569595, -53.2685738, 43.0400238, -95.6208038, 95.9255371
1: -31.1129627, 35.7509232, -31.5789700, 36.0733871, -67.1863480, 67.3298950
2: -29.7831135, 35.2280579, -30.3186092, 35.6439133, -65.4270248, 65.5466690
3: -33.2266235, 41.1012955, -33.8265076, 41.6299706, -74.8565979, 74.9277954
4: -39.4062729, 38.5204010, -39.9514847, 38.9526215, -78.3588943, 78.4718857
5: -36.1926537, 40.8908997, -36.7789154, 41.4116364, -77.6042862, 77.6698151
6: -55.6681213, 22.1476135, -55.9515228, 22.4241123, -78.0922241, 78.0991364
7: -42.2255020, 39.8046188, -42.8355255, 40.2237473, -82.4492416, 82.6401367
8: -38.6507568, 45.0255814, -39.2809753, 45.5468750, -84.1976318, 84.3065567
9: -33.6869049, 37.2987442, -34.1249123, 37.5144386, -71.2013397, 71.4236603
10: -54.7885666, 51.8815689, -55.2027397, 52.2972603, -107.0858231, 107.0843048
11: -56.0268784, 39.1437759, -56.5374298, 39.6088257, -95.6357040, 95.6812057
12: -58.7011070, 43.3406601, -59.1971741, 43.9291763, -102.6302795, 102.5378342
13: -48.0890503, 49.0989990, -48.6337357, 49.6029396, -97.6919861, 97.7327347
14: -80.8339844, 42.9068146, -81.5054855, 43.3417511, -124.1757355, 124.4122849
15: -39.9216995, 36.1562271, -40.3365250, 36.3883553, -76.3100586, 76.4927444
16: -57.7627029, 40.6961670, -58.2324524, 40.8659668, -98.6286697, 98.9286194
17: -84.7511139, 62.1553497, -85.2160034, 62.5052032, -147.2563171, 147.3713379
18: -48.5824661, 28.4321404, -49.0285378, 28.9954910, -77.5779572, 77.4606781
19: -41.0186501, 19.0454330, -41.4202881, 19.4186325, -60.4372787, 60.4657211
20: -35.1311798, 21.3704300, -35.4192963, 21.7251720, -56.8563423, 56.7897263
21: -48.7951202, 24.9908028, -49.2218323, 25.3763752, -74.1714935, 74.2126312
22: -50.5590210, 29.4732456, -51.0241013, 29.9840126, -80.5430298, 80.4973450
23: -38.7067528, 26.0304642, -39.2091980, 26.4918346, -65.1985779, 65.2396622
24: -44.7379913, 22.3332500, -45.2751465, 22.7585754, -67.4965668, 67.6083984
25: -38.1128616, 30.4168682, -38.5636406, 30.9360847, -69.0489502, 68.9805069
26: -58.5407181, 36.6361732, -59.1455307, 37.4554977, -95.9962158, 95.7817078
27: -48.9253578, 26.8304710, -49.4031487, 27.2447548, -76.1701050, 76.2336197
28: -37.4990921, 28.2882233, -37.9060669, 28.7408772, -66.2399673, 66.1942902
29: -54.9600639, 33.7991447, -55.5003014, 34.2723389, -89.2324066, 89.2994385
30: -47.3537521, 26.7802544, -47.8243828, 27.1716633, -74.5254135, 74.6046371
31: -48.4992561, 23.5551224, -49.0603142, 23.9456272, -72.4448853, 72.6154327
32: -48.7989769, 27.0678654, -49.1726799, 27.3904438, -76.1894226, 76.2405472
33: -71.4085464, 43.6503029, -71.8656769, 44.0348930, -115.4434357, 115.5159760
34: -60.5449448, 29.4392471, -60.9410362, 29.9493504, -90.4942856, 90.3802795
35: -56.8118515, 34.2967987, -57.2161331, 34.6732788, -91.4851227, 91.5129242
36: -56.9169731, 33.4024277, -57.3017159, 33.8629990, -90.7799683, 90.7041397
37: -84.6375427, 32.4511871, -85.2896576, 33.0243912, -117.6619263, 117.7408447
38: -68.6771545, 40.3593826, -69.1105042, 40.8608093, -109.5379486, 109.4698715
39: -84.5344086, 40.4171104, -85.0271149, 40.7392235, -125.2736359, 125.4442215
40: -74.8639221, 29.7453575, -75.2194672, 29.9860611, -104.8499832, 104.9648209
41: -53.9326057, 25.4611549, -54.3450356, 25.8963242, -79.8289337, 79.8061829
42: -38.7033005, 28.9884186, -38.9586754, 29.3561325, -68.0594330, 67.9470978

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=204, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=403, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 645
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 976

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 729

## Relational analysis of IS_A1_A1_A1_A1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_A1_A1_A1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.6244522, upper bound: 38.8820855
time: 75.63 seconds

## Relational analysis of IS_A1_A1_A1_A1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_A1_A1_A1_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.6244522, upper bound: 38.9169427
time: 73.75 seconds

## BFS IS instance: IS_A1_A1_A1_A1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -52.4893913, 42.6311340, -53.2928772, 43.0449829, -95.5343552, 95.9240036
1: -31.0440922, 35.7167740, -31.5944729, 36.0782585, -67.1223450, 67.3112488
2: -29.7513504, 35.2386360, -30.3485222, 35.6499596, -65.4013062, 65.5871506
3: -33.1832123, 41.1216965, -33.8645592, 41.6382523, -74.8214645, 74.9862518
4: -39.3979874, 38.5296822, -39.9817619, 38.9593201, -78.3572998, 78.5114365
5: -36.1515884, 40.9069138, -36.8168564, 41.4204330, -77.5720062, 77.7237701
6: -55.6838646, 22.1170845, -55.9652519, 22.4351444, -78.1190033, 78.0823364
7: -42.1290436, 39.7478180, -42.8623505, 40.2298050, -82.3588486, 82.6101685
8: -38.5460663, 44.9632797, -39.2992325, 45.5577736, -84.1038361, 84.2625122
9: -33.6999817, 37.2766037, -34.1453362, 37.5174751, -71.2174530, 71.4219360
10: -54.8186569, 51.9715919, -55.2170181, 52.3415298, -107.1601868, 107.1886139
11: -56.0668526, 39.2226105, -56.5456619, 39.6506081, -95.7174606, 95.7682724
12: -58.6982117, 43.2664299, -59.2075233, 43.9581299, -102.6563416, 102.4739532
13: -48.2408371, 49.2241631, -48.6995087, 49.6138115, -97.8546371, 97.9236603
14: -80.7404099, 42.9211884, -81.5237122, 43.3786774, -124.1190872, 124.4449005
15: -39.8739700, 36.1533012, -40.3529320, 36.3965607, -76.2705307, 76.5062256
16: -57.7691345, 40.6536560, -58.2517624, 40.8694229, -98.6385574, 98.9054108
17: -84.6634369, 62.1689262, -85.2306442, 62.5233231, -147.1867523, 147.3995667
18: -48.5960159, 28.5051422, -49.0404663, 29.0452538, -77.6412659, 77.5456085
19: -41.0415802, 19.0836258, -41.4288025, 19.4446220, -60.4861984, 60.5124283
20: -35.1483040, 21.4146290, -35.4270172, 21.7494507, -56.8977470, 56.8416443
21: -48.8341599, 25.0598831, -49.2320862, 25.4069519, -74.2411118, 74.2919693
22: -50.5536270, 29.4896488, -51.0351677, 30.0118847, -80.5655060, 80.5248184
23: -38.7584381, 26.0991993, -39.2166977, 26.5284691, -65.2869110, 65.3158875
24: -44.7683601, 22.3858852, -45.2842636, 22.7914162, -67.5597763, 67.6701508
25: -38.1381569, 30.4427357, -38.5727463, 30.9662495, -69.1044006, 69.0154800
26: -58.5418625, 36.6409035, -59.1547737, 37.5108681, -96.0527344, 95.7956696
27: -48.9958611, 26.9411964, -49.4150772, 27.2904682, -76.2863235, 76.3562775
28: -37.5359650, 28.3427448, -37.9131012, 28.7730236, -66.3089905, 66.2558441
29: -54.9545860, 33.8584709, -55.5114136, 34.3074188, -89.2620087, 89.3698883
30: -47.3697014, 26.8224201, -47.8318062, 27.2012749, -74.5709763, 74.6542206
31: -48.5574799, 23.6265507, -49.0736656, 23.9742508, -72.5317307, 72.7002182
32: -48.8233833, 27.0328255, -49.1856194, 27.4030380, -76.2264252, 76.2184448
33: -71.4467316, 43.5717926, -71.9068069, 44.0436668, -115.4903870, 115.4785995
34: -60.5171814, 29.3987617, -60.9529343, 29.9734917, -90.4906769, 90.3516922
35: -56.8271713, 34.2306938, -57.2490540, 34.6820602, -91.5092316, 91.4797516
36: -56.9157410, 33.2962799, -57.3228073, 33.8697701, -90.7855072, 90.6190872
37: -84.5589600, 32.2677879, -85.3095322, 33.0366287, -117.5955658, 117.5773163
38: -68.6876526, 40.3203697, -69.1321411, 40.8759308, -109.5635757, 109.4525070
39: -84.5726929, 40.3290329, -85.0653000, 40.7444916, -125.3171692, 125.3943329
40: -74.8392181, 29.6587429, -75.2348328, 29.9944000, -104.8336182, 104.8935776
41: -53.9287300, 25.4036007, -54.3570023, 25.9160576, -79.8447723, 79.7606049
42: -38.7365341, 29.0061798, -38.9658890, 29.3801994, -68.1167297, 67.9720688

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=204, inp2_unstable=209, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=401, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1491
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 1282

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 713

## Relational analysis of IS_A1_A1_A1_A1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_A1_A1_A1_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.6202687, upper bound: 38.9297793
time: 71.46 seconds

## Relational analysis of IS_A1_A1_A1_A1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_A1_A1_A1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.6542580, upper bound: 38.9815413
time: 64.21 seconds

## BFS IS instance: IS_A1_A1_A1_A1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -52.6788216, 42.6950607, -53.2928772, 43.0449829, -95.7238007, 95.9879379
1: -31.1666050, 35.7769356, -31.5944729, 36.0782585, -67.2448578, 67.3714066
2: -29.8796806, 35.2999039, -30.3485222, 35.6499596, -65.5296402, 65.6484222
3: -33.3450317, 41.2110062, -33.8645592, 41.6382523, -74.9832840, 75.0755615
4: -39.5063210, 38.5851479, -39.9817619, 38.9593201, -78.4656372, 78.5668945
5: -36.3108749, 40.9936676, -36.8168564, 41.4204330, -77.7313080, 77.8105240
6: -55.7291107, 22.1885700, -55.9652519, 22.4351444, -78.1642532, 78.1538239
7: -42.3107452, 39.8479385, -42.8623505, 40.2298050, -82.5405426, 82.7102814
8: -38.7158813, 45.0848312, -39.2992325, 45.5577736, -84.2736511, 84.3840637
9: -33.7608452, 37.3169098, -34.1453362, 37.5174751, -71.2783203, 71.4622421
10: -54.8897476, 52.0252609, -55.2170181, 52.3415298, -107.2312775, 107.2422791
11: -56.1316872, 39.2719879, -56.5456619, 39.6506081, -95.7822876, 95.8176498
12: -58.7907181, 43.4362259, -59.2075233, 43.9581299, -102.7488403, 102.6437531
13: -48.2939301, 49.2686996, -48.6995087, 49.6138115, -97.9077454, 97.9682007
14: -80.9642944, 43.0214386, -81.5237122, 43.3786774, -124.3429718, 124.5451431
15: -39.9800301, 36.1953659, -40.3529320, 36.3965607, -76.3765869, 76.5482941
16: -57.8386421, 40.7083969, -58.2517624, 40.8694229, -98.7080688, 98.9601517
17: -84.8107452, 62.2306290, -85.2306442, 62.5233231, -147.3340607, 147.4612732
18: -48.6862488, 28.5888386, -49.0404663, 29.0452538, -77.7314987, 77.6293030
19: -41.0999146, 19.1238899, -41.4288025, 19.4446220, -60.5445328, 60.5526924
20: -35.1890564, 21.4517956, -35.4270172, 21.7494507, -56.9385033, 56.8788147
21: -48.8856239, 25.0849514, -49.2320862, 25.4069519, -74.2925720, 74.3170395
22: -50.6449127, 29.5611877, -51.0351677, 30.0118847, -80.6567993, 80.5963516
23: -38.8098869, 26.1444893, -39.2166977, 26.5284691, -65.3383560, 65.3611908
24: -44.8500519, 22.4373302, -45.2842636, 22.7914162, -67.6414642, 67.7215958
25: -38.1944351, 30.5134468, -38.5727463, 30.9662495, -69.1606827, 69.0861969
26: -58.6488533, 36.8146744, -59.1547737, 37.5108681, -96.1597214, 95.9694366
27: -49.0608025, 26.9703960, -49.4150772, 27.2904682, -76.3512726, 76.3854675
28: -37.5840034, 28.3937225, -37.9131012, 28.7730236, -66.3570251, 66.3068237
29: -55.0686951, 33.9088860, -55.5114136, 34.3074188, -89.3761139, 89.4202957
30: -47.4361496, 26.8748646, -47.8318062, 27.2012749, -74.6374207, 74.7066727
31: -48.6240501, 23.6425171, -49.0736656, 23.9742508, -72.5982971, 72.7161865
32: -48.8767929, 27.1087837, -49.1856194, 27.4030380, -76.2798309, 76.2944031
33: -71.5429535, 43.7109795, -71.9068069, 44.0436668, -115.5866241, 115.6177826
34: -60.6000977, 29.5232353, -60.9529343, 29.9734917, -90.5735779, 90.4761658
35: -56.9218140, 34.3516998, -57.2490540, 34.6820602, -91.6038742, 91.6007538
36: -56.9979019, 33.4274139, -57.3228073, 33.8697701, -90.8676758, 90.7502213
37: -84.7281799, 32.4996796, -85.3095322, 33.0366287, -117.7648087, 117.8092041
38: -68.7593231, 40.4233627, -69.1321411, 40.8759308, -109.6352463, 109.5555038
39: -84.6807785, 40.4603386, -85.0653000, 40.7444916, -125.4252625, 125.5256348
40: -74.9347458, 29.7783928, -75.2348328, 29.9944000, -104.9291382, 105.0132294
41: -54.0050964, 25.5303745, -54.3570023, 25.9160576, -79.9211578, 79.8873749
42: -38.7582626, 29.0687675, -38.9658890, 29.3801994, -68.1384583, 68.0346527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=204, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=403, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 645
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 976

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 729

## Relational analysis of IS_A1_A1_A1_A1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_A1_A1_A1_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.6574702, upper bound: 38.8836060
time: 71.02 seconds

## Relational analysis of IS_A1_A1_A1_A1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_A1_A1_A1_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.6574702, upper bound: 38.9184875
time: 71.35 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -52.8847008, 42.8136292, -53.4187393, 43.0612564, -95.9459534, 96.2323685
1: -31.3098660, 35.8731041, -31.6890011, 36.0919952, -67.4018631, 67.5621033
2: -30.0169277, 35.3787308, -30.4425564, 35.6618538, -65.6787796, 65.8212891
3: -33.4903336, 41.2980576, -33.9621201, 41.6553841, -75.1457214, 75.2601776
4: -39.6684952, 38.6855049, -40.0776062, 38.9721603, -78.6406555, 78.7631073
5: -36.4439774, 41.0878448, -36.9089584, 41.4384117, -77.8823853, 77.9968033
6: -55.7795448, 22.3464146, -55.9767838, 22.5091839, -78.2887268, 78.3231964
7: -42.4594498, 39.9363251, -42.9787903, 40.2469025, -82.7063522, 82.9151154
8: -38.9028625, 45.2017593, -39.4351997, 45.5736198, -84.4764709, 84.6369629
9: -33.9290390, 37.3845825, -34.2238426, 37.5384064, -71.4674377, 71.6084290
10: -54.9440308, 52.0893784, -55.2651634, 52.3721352, -107.3161621, 107.3545380
11: -56.2187271, 39.4137001, -56.5807228, 39.7189865, -95.9377060, 95.9944229
12: -58.9102173, 43.6439896, -59.2241592, 44.0734482, -102.9836578, 102.8681488
13: -48.3758163, 49.2835197, -48.7374039, 49.6461945, -98.0220108, 98.0209122
14: -81.0198212, 43.0448380, -81.6289368, 43.3921280, -124.4119415, 124.6737747
15: -40.1692276, 36.2547531, -40.4494133, 36.4037323, -76.5729523, 76.7041626
16: -58.0042686, 40.7559814, -58.3120651, 40.8805618, -98.8848267, 99.0680389
17: -84.9405975, 62.2329025, -85.3008270, 62.5252647, -147.4658508, 147.5337219
18: -48.7421875, 28.7508793, -49.0765800, 29.1194592, -77.8616409, 77.8274612
19: -41.1520691, 19.2282333, -41.4485054, 19.5042877, -60.6563454, 60.6767387
20: -35.2672234, 21.5781155, -35.4476547, 21.8055954, -57.0728149, 57.0257721
21: -48.9599571, 25.1913662, -49.2616119, 25.4613514, -74.4213104, 74.4529724
22: -50.7063751, 29.6804886, -51.0759468, 30.0993404, -80.8057175, 80.7564392
23: -38.8773537, 26.2562637, -39.2392387, 26.5891094, -65.4664612, 65.4954987
24: -44.9029160, 22.5207157, -45.3169289, 22.8463097, -67.7492218, 67.8376465
25: -38.2571411, 30.6283665, -38.6039047, 31.0488911, -69.3060303, 69.2322693
26: -58.7560883, 37.0289688, -59.1933937, 37.6477394, -96.4038239, 96.2223663
27: -49.1254616, 27.1001110, -49.4403992, 27.3441219, -76.4695816, 76.5405121
28: -37.6241608, 28.5037956, -37.9350052, 28.8350601, -66.4592209, 66.4387970
29: -55.1212158, 33.9948540, -55.5528831, 34.3781471, -89.4993591, 89.5477371
30: -47.5109520, 26.9720058, -47.8722305, 27.2544041, -74.7653580, 74.8442383
31: -48.7017517, 23.7684059, -49.1020164, 24.0326195, -72.7343750, 72.8704224
32: -48.9830818, 27.3142967, -49.1995163, 27.4923592, -76.4754410, 76.5138092
33: -71.5056152, 43.7359238, -71.8777695, 44.1099625, -115.6155777, 115.6136856
34: -60.6874008, 29.6793308, -60.9779091, 30.0659180, -90.7533112, 90.6572418
35: -56.9137268, 34.4309082, -57.2408409, 34.7528839, -91.6666031, 91.6717453
36: -57.0788422, 33.6699791, -57.3322411, 33.9976501, -91.0764923, 91.0022202
37: -84.8705292, 32.6577835, -85.3449326, 33.1609230, -118.0314484, 118.0027084
38: -68.9022827, 40.7321510, -69.1546936, 40.9938126, -109.8960953, 109.8868408
39: -84.7324219, 40.5573425, -85.0803070, 40.8081818, -125.5405884, 125.6376495
40: -75.0187683, 29.8695564, -75.2646637, 30.0441723, -105.0629349, 105.1342163
41: -54.1057129, 25.7081261, -54.3726425, 26.0081654, -80.1138763, 80.0807648
42: -38.8081436, 29.1995888, -38.9776230, 29.4434128, -68.2515488, 68.1772079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=207, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=404, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1491
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 976

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 727

## Relational analysis of IS_A1_A2_B2_A1_B2_B2_A1_A1_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1_B2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.7157240, upper bound: 38.9879779
time: 84.98 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2_B2_A1_A1_B2_A2

### Relational analysis result of IS_A1_A2_B2_A1_B2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.7157240, upper bound: 38.9898836
time: 73.82 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -52.9853020, 42.8523102, -53.4432335, 43.0663834, -96.0516815, 96.2955399
1: -31.3648338, 35.8996887, -31.7045612, 36.0970154, -67.4618454, 67.6042480
2: -30.1144295, 35.4512062, -30.4725246, 35.6680527, -65.7824860, 65.9237289
3: -33.6099701, 41.4080963, -34.0002518, 41.6637878, -75.2737579, 75.4083481
4: -39.7694321, 38.7511749, -40.1079369, 38.9790306, -78.7484589, 78.8591156
5: -36.5635529, 41.1910706, -36.9469452, 41.4473190, -78.0108719, 78.1380157
6: -55.8423729, 22.3891258, -55.9907074, 22.5203323, -78.3627014, 78.3798370
7: -42.5461159, 39.9801788, -43.0056953, 40.2531013, -82.7992096, 82.9858704
8: -38.9690971, 45.2621689, -39.4534836, 45.5848236, -84.5539246, 84.7156525
9: -34.0044022, 37.4036179, -34.2444344, 37.5415955, -71.5459900, 71.6480560
10: -55.0447845, 52.2351761, -55.2795486, 52.4166336, -107.4614105, 107.5147247
11: -56.3254204, 39.5443840, -56.5891571, 39.7608604, -96.0862808, 96.1335373
12: -59.0008011, 43.7411919, -59.2346878, 44.1025620, -103.1033630, 102.9758759
13: -48.5826836, 49.4541855, -48.8035431, 49.6573448, -98.2400208, 98.2577286
14: -81.1534882, 43.1635284, -81.6473694, 43.4294281, -124.5829163, 124.8108978
15: -40.2287598, 36.2944145, -40.4659653, 36.4121399, -76.6408997, 76.7603760
16: -58.0820770, 40.7691116, -58.3315849, 40.8841705, -98.9662476, 99.1006927
17: -85.0037460, 62.3105316, -85.3157196, 62.5448608, -147.5485992, 147.6262512
18: -48.8470955, 28.9092331, -49.0887871, 29.1696167, -78.0167084, 77.9980164
19: -41.2344589, 19.3072643, -41.4571686, 19.5303078, -60.7647667, 60.7644348
20: -35.3263779, 21.6602898, -35.4555511, 21.8300171, -57.1563950, 57.1158409
21: -49.0521660, 25.2859879, -49.2720718, 25.4919624, -74.5441208, 74.5580597
22: -50.7942581, 29.7689476, -51.0872574, 30.1272697, -80.9215240, 80.8562012
23: -38.9814301, 26.3708916, -39.2468834, 26.6258392, -65.6072693, 65.6177750
24: -45.0160370, 22.6257343, -45.3262901, 22.8792896, -67.8953247, 67.9520187
25: -38.3399277, 30.7256069, -38.6131935, 31.0791893, -69.4191132, 69.3387985
26: -58.8652611, 37.2090073, -59.2028732, 37.7032547, -96.5685120, 96.4118805
27: -49.2620926, 27.2410736, -49.4526176, 27.3899899, -76.6520767, 76.6936874
28: -37.7104111, 28.6099930, -37.9422913, 28.8672943, -66.5777054, 66.5522842
29: -55.2319565, 34.1057739, -55.5642891, 34.4133530, -89.6453094, 89.6700592
30: -47.5954628, 27.0675068, -47.8799896, 27.2841759, -74.8796387, 74.9474945
31: -48.8278275, 23.8565502, -49.1155396, 24.0613403, -72.8891678, 72.9720917
32: -49.0620537, 27.3565998, -49.2126694, 27.5051155, -76.5671692, 76.5692673
33: -71.6488953, 43.7988586, -71.9204788, 44.1188393, -115.7677307, 115.7193375
34: -60.7435913, 29.7645683, -60.9900055, 30.0901661, -90.8337479, 90.7545776
35: -57.0277786, 34.4864960, -57.2743149, 34.7617149, -91.7894897, 91.7608032
36: -57.1597252, 33.6956787, -57.3537521, 34.0043945, -91.1641083, 91.0494308
37: -84.9624710, 32.7071419, -85.3652191, 33.1732483, -118.1357193, 118.0723495
38: -68.9852905, 40.7977943, -69.1766891, 41.0092850, -109.9945679, 109.9744873
39: -84.8811417, 40.6018944, -85.1191864, 40.8135071, -125.6946487, 125.7210846
40: -75.0907593, 29.9037628, -75.2803345, 30.0525742, -105.1433334, 105.1840973
41: -54.1789703, 25.7786751, -54.3848419, 26.0280037, -80.2069702, 80.1635132
42: -38.8648911, 29.2813950, -38.9849548, 29.4680023, -68.3328934, 68.2663498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=206, inp2_unstable=207, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=406, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 727

## Relational analysis of IS_A1_A2_B2_A1_B2_B2_A2_A1_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1_B2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.7566663, upper bound: 38.9900691
time: 77.08 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2_B2_A2_A1_B2_A2

### Relational analysis result of IS_A1_A2_B2_A1_B2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.7566663, upper bound: 38.9920083
time: 122.59 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 202.14 seconds
IS_A1_A1_A1_A1_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 202.14
Output dim: 2, lower bound: -38.6202687, upper bound: 38.9282816
IS_A1_A1_A1_A1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 202.14
Output dim: 2, lower bound: -38.6202687, upper bound: 38.9800464
IS_A1_A1_A1_A1_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 202.14
Output dim: 2, lower bound: -38.6244522, upper bound: 38.8820855
IS_A1_A1_A1_A1_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 202.14
Output dim: 2, lower bound: -38.6244522, upper bound: 38.9169427
IS_A1_A1_A1_A1_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 202.14
Output dim: 2, lower bound: -38.6202687, upper bound: 38.9297793
IS_A1_A1_A1_A1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 202.14
Output dim: 2, lower bound: -38.6542580, upper bound: 38.9815413
IS_A1_A1_A1_A1_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 202.14
Output dim: 2, lower bound: -38.6574702, upper bound: 38.8836060
IS_A1_A1_A1_A1_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 202.14
Output dim: 2, lower bound: -38.6574702, upper bound: 38.9184875
IS_A1_A2_B2_A1_B2_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 202.14
Output dim: 2, lower bound: -38.7157240, upper bound: 38.9879779
IS_A1_A2_B2_A1_B2_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 202.14
Output dim: 2, lower bound: -38.7157240, upper bound: 38.9898836
IS_A1_A2_B2_A1_B2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 202.14
Output dim: 2, lower bound: -38.7566663, upper bound: 38.9900691
IS_A1_A2_B2_A1_B2_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 202.14
Output dim: 2, lower bound: -38.7566663, upper bound: 38.9920083

## BFS IS instance: IS_A1_A1_A1_A1_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -52.3879204, 42.5908203, -53.2594757, 43.0341759, -95.4220963, 95.8502960
1: -30.9882584, 35.6890488, -31.5733624, 36.0687408, -67.0569916, 67.2624054
2: -29.6527576, 35.1647034, -30.3130398, 35.6381950, -65.2909470, 65.4777374
3: -33.0626183, 41.0105972, -33.8206863, 41.6261864, -74.6888046, 74.8312836
4: -39.2961922, 38.4632187, -39.9468384, 38.9478951, -78.2440796, 78.4100571
5: -36.0316200, 40.8025932, -36.7741318, 41.4072876, -77.4389038, 77.5767212
6: -55.6136551, 22.0734558, -55.9239540, 22.4168186, -78.0304718, 77.9974060
7: -42.0397682, 39.7014923, -42.8256760, 40.2155151, -82.2552795, 82.5271606
8: -38.4783707, 44.9020844, -39.2739983, 45.5414276, -84.0197983, 84.1760788
9: -33.6247025, 37.2555008, -34.1213188, 37.5056610, -71.1303635, 71.3768158
10: -54.7151566, 51.8260422, -55.1963043, 52.2919044, -107.0070648, 107.0223389
11: -55.9529953, 39.0921326, -56.5104866, 39.6027985, -95.5557938, 95.6026154
12: -58.6060486, 43.1688156, -59.1900787, 43.9237518, -102.5298004, 102.3588943
13: -48.0329323, 49.0398521, -48.6253433, 49.5629311, -97.5958557, 97.6651917
14: -80.6056976, 42.8050308, -81.4932861, 43.3373184, -123.9430161, 124.2983170
15: -39.8127213, 36.1019897, -40.3289528, 36.3516579, -76.1643753, 76.4309387
16: -57.6855583, 40.6389580, -58.2118759, 40.8590546, -98.5446167, 98.8508301
17: -84.6004639, 62.0890388, -85.2070923, 62.4918900, -147.0923462, 147.2961273
18: -48.4869652, 28.3464546, -49.0127335, 28.9900417, -77.4770050, 77.3591919
19: -40.9591217, 19.0039577, -41.4169846, 19.4152107, -60.3743248, 60.4209404
20: -35.0882645, 21.3317184, -35.4132881, 21.7208691, -56.8091354, 56.7450066
21: -48.7417793, 24.9642601, -49.2165794, 25.3722076, -74.1139832, 74.1808319
22: -50.4640617, 29.3954010, -51.0139771, 29.9687939, -80.4328461, 80.4093781
23: -38.6538544, 25.9838676, -39.2050018, 26.4881668, -65.1420212, 65.1888657
24: -44.6541214, 22.2804413, -45.2690277, 22.7548447, -67.4089661, 67.5494690
25: -38.0532074, 30.3404903, -38.5544281, 30.9221725, -68.9753799, 68.8949203
26: -58.4314537, 36.4581604, -59.1394653, 37.4440422, -95.8754959, 95.5976257
27: -48.8540688, 26.7998486, -49.3847313, 27.2407494, -76.0948181, 76.1845779
28: -37.4501419, 28.2352066, -37.9037132, 28.7350445, -66.1851807, 66.1389160
29: -54.8427010, 33.7453117, -55.4914474, 34.2636414, -89.1063385, 89.2367554
30: -47.2849426, 26.7263145, -47.8177948, 27.1675358, -74.4524689, 74.5440979
31: -48.4304008, 23.5374584, -49.0541801, 23.9411011, -72.3714981, 72.5916367
32: -48.7403069, 26.9903126, -49.1579094, 27.3863354, -76.1266327, 76.1482239
33: -71.3081512, 43.5085449, -71.8537445, 44.0279541, -115.3361053, 115.3622894
34: -60.4590988, 29.3132095, -60.9330254, 29.9449863, -90.4040756, 90.2462311
35: -56.7138214, 34.1738014, -57.2066574, 34.6679535, -91.3817749, 91.3804626
36: -56.8325768, 33.2687073, -57.2955856, 33.8560181, -90.6885910, 90.5642853
37: -84.4649734, 32.2165489, -85.2802277, 33.0171661, -117.4821320, 117.4967804
38: -68.5992508, 40.2551651, -69.0930634, 40.8576279, -109.4568787, 109.3482285
39: -84.4222336, 40.2837219, -85.0155029, 40.7340393, -125.1562653, 125.2992172
40: -74.7637177, 29.6225548, -75.2051010, 29.9777966, -104.7415009, 104.8276520
41: -53.8501167, 25.3314095, -54.3261948, 25.8882923, -79.7384033, 79.6576080
42: -38.6757736, 28.9235344, -38.9412003, 29.3499451, -68.0257111, 67.8647308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=204, inp2_unstable=208, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=401, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 1282

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 729

## Relational analysis of IS_A1_A1_A1_A1_A1_B2_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_A1_A1_A1_A1_B2_A1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.6202687, upper bound: 38.9456614
time: 83.39 seconds

## Relational analysis of IS_A1_A1_A1_A1_A1_B2_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_A1_A1_A1_A1_B2_A1_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.6202687, upper bound: 38.9282819
time: 78.22 seconds

## BFS IS instance: IS_A1_A1_A1_A1_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -52.4859848, 42.6289291, -53.2837601, 43.0391045, -95.5250854, 95.9126892
1: -31.0419083, 35.7150650, -31.5888481, 36.0736389, -67.1155472, 67.3039093
2: -29.7493210, 35.2365570, -30.3429470, 35.6442337, -65.3935547, 65.5794983
3: -33.1810226, 41.1203194, -33.8587418, 41.6344910, -74.8155136, 74.9790649
4: -39.3962860, 38.5279770, -39.9771080, 38.9545898, -78.3508606, 78.5050812
5: -36.1498413, 40.9053574, -36.8120728, 41.4160995, -77.5659332, 77.7174301
6: -55.6746140, 22.1143913, -55.9376564, 22.4278412, -78.1024323, 78.0520477
7: -42.1250153, 39.7448120, -42.8525009, 40.2215767, -82.3465881, 82.5973053
8: -38.5435219, 44.9613495, -39.2922134, 45.5523415, -84.0958633, 84.2535629
9: -33.6986160, 37.2736740, -34.1417389, 37.5086975, -71.2073135, 71.4154129
10: -54.8164368, 51.9697380, -55.2105980, 52.3361588, -107.1525955, 107.1803360
11: -56.0577812, 39.2203789, -56.5186958, 39.6445999, -95.7023773, 95.7390747
12: -58.6956863, 43.2644157, -59.2004280, 43.9527054, -102.6483917, 102.4648438
13: -48.2378082, 49.2095757, -48.6911163, 49.5738602, -97.8116608, 97.9006958
14: -80.7360077, 42.9196472, -81.5115051, 43.3742332, -124.1102371, 124.4311523
15: -39.8710861, 36.1411018, -40.3453674, 36.3598824, -76.2309723, 76.4864655
16: -57.7614861, 40.6511688, -58.2312164, 40.8625183, -98.6240082, 98.8823853
17: -84.6600876, 62.1643448, -85.2217178, 62.5099640, -147.1700439, 147.3860626
18: -48.5908012, 28.5031528, -49.0246582, 29.0398178, -77.6306152, 77.5278015
19: -41.0403862, 19.0824203, -41.4254837, 19.4411850, -60.4815674, 60.5079041
20: -35.1461449, 21.4130630, -35.4210243, 21.7451439, -56.8912888, 56.8340836
21: -48.8322830, 25.0584145, -49.2268257, 25.4027729, -74.2350540, 74.2852402
22: -50.5499725, 29.4833393, -51.0250473, 29.9966469, -80.5466156, 80.5083847
23: -38.7570190, 26.0978813, -39.2124710, 26.5248051, -65.2818146, 65.3103485
24: -44.7661705, 22.3845482, -45.2781525, 22.7876854, -67.5538559, 67.6627045
25: -38.1347733, 30.4370728, -38.5635300, 30.9523182, -69.0870819, 69.0006027
26: -58.5395699, 36.6366348, -59.1487083, 37.4994469, -96.0390167, 95.7853394
27: -48.9895401, 26.9397812, -49.3966522, 27.2864704, -76.2760086, 76.3364258
28: -37.5350761, 28.3407288, -37.9107513, 28.7671604, -66.3022385, 66.2514801
29: -54.9513321, 33.8550453, -55.5025444, 34.2986908, -89.2500153, 89.3575897
30: -47.3673096, 26.8209305, -47.8252106, 27.1971474, -74.5644531, 74.6461411
31: -48.5552292, 23.6248779, -49.0675316, 23.9697342, -72.5249634, 72.6924057
32: -48.8181038, 27.0312538, -49.1708565, 27.3989258, -76.2170258, 76.2021027
33: -71.4425125, 43.5692978, -71.8948822, 44.0367279, -115.4792404, 115.4641800
34: -60.5143356, 29.3972130, -60.9449158, 29.9690895, -90.4834213, 90.3421249
35: -56.8237381, 34.2287331, -57.2395821, 34.6766815, -91.5004196, 91.4683151
36: -56.9135246, 33.2937279, -57.3166809, 33.8628006, -90.7763214, 90.6104126
37: -84.5555801, 32.2649918, -85.3001099, 33.0294342, -117.5850143, 117.5651016
38: -68.6814194, 40.3191566, -69.1146927, 40.8727875, -109.5542068, 109.4338379
39: -84.5686035, 40.3269272, -85.0536728, 40.7392616, -125.3078613, 125.3806000
40: -74.8344574, 29.6556206, -75.2204895, 29.9861298, -104.8205872, 104.8760986
41: -53.9225845, 25.4005833, -54.3381615, 25.9080162, -79.8305969, 79.7387390
42: -38.7307243, 29.0039062, -38.9484024, 29.3740025, -68.1047287, 67.9523087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=204, inp2_unstable=208, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=401, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1491
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 1282

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 729

## Relational analysis of IS_A1_A1_A1_A1_A2_B2_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_A1_A1_A1_A2_B2_A1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.6542580, upper bound: 38.9471690
time: 79.83 seconds

## Relational analysis of IS_A1_A1_A1_A1_A2_B2_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_A1_A1_A1_A2_B2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.6542580, upper bound: 38.9815413
time: 128.04 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B2_B2_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -52.6886139, 42.7477036, -53.4187393, 43.0612564, -95.7498627, 96.1664429
1: -31.1829376, 35.8115349, -31.6890011, 36.0919952, -67.2749329, 67.5005341
2: -29.8846493, 35.3157349, -30.4425564, 35.6618538, -65.5465012, 65.7582855
3: -33.3233795, 41.2063065, -33.9621201, 41.6553841, -74.9787598, 75.1684265
4: -39.5557404, 38.6284790, -40.0776062, 38.9721603, -78.5279007, 78.7060852
5: -36.2800980, 40.9980698, -36.9089584, 41.4384117, -77.7185059, 77.9070282
6: -55.7320023, 22.2710381, -55.9767838, 22.5091839, -78.2411804, 78.2478180
7: -42.2717133, 39.8322639, -42.9787903, 40.2469025, -82.5186081, 82.8110504
8: -38.7279968, 45.0775299, -39.4351997, 45.5736198, -84.3016205, 84.5127258
9: -33.8649673, 37.3369179, -34.2238426, 37.5384064, -71.4033737, 71.5607605
10: -54.8667374, 52.0324402, -55.2651634, 52.3721352, -107.2388763, 107.2976074
11: -56.1459236, 39.3605270, -56.5807228, 39.7189865, -95.8649063, 95.9412460
12: -58.8140640, 43.4688873, -59.2241592, 44.0734482, -102.8875122, 102.6930466
13: -48.3185730, 49.2239685, -48.7374039, 49.6461945, -97.9647675, 97.9613647
14: -80.7884293, 42.9425087, -81.6289368, 43.3921280, -124.1805573, 124.5714417
15: -40.0581093, 36.2089539, -40.4494133, 36.4037323, -76.4618378, 76.6583710
16: -57.9186935, 40.6974640, -58.3120651, 40.8805618, -98.7992554, 99.0095215
17: -84.7822723, 62.1679840, -85.3008270, 62.5252647, -147.3075409, 147.4688110
18: -48.6491699, 28.6637802, -49.0765800, 29.1194592, -77.7686310, 77.7403564
19: -41.0915337, 19.1857681, -41.4485054, 19.5042877, -60.5958138, 60.6342735
20: -35.2163887, 21.5384521, -35.4476547, 21.8055954, -57.0219841, 56.9861069
21: -48.9006920, 25.1638031, -49.2616119, 25.4613514, -74.3620453, 74.4253998
22: -50.6092567, 29.6040859, -51.0759468, 30.0993404, -80.7085953, 80.6800308
23: -38.8240814, 26.2084312, -39.2392387, 26.5891094, -65.4131927, 65.4476624
24: -44.8181648, 22.4667435, -45.3169289, 22.8463097, -67.6644745, 67.7836685
25: -38.1958923, 30.5547466, -38.6039047, 31.0488911, -69.2447815, 69.1586533
26: -58.6450615, 36.8499336, -59.1933937, 37.6477394, -96.2928009, 96.0433273
27: -49.0477982, 27.0685444, -49.4403992, 27.3441219, -76.3919220, 76.5089417
28: -37.5738602, 28.4507217, -37.9350052, 28.8350601, -66.4089203, 66.3857269
29: -54.9987183, 33.9412918, -55.5528831, 34.3781471, -89.3768616, 89.4941711
30: -47.4415550, 26.9168034, -47.8722305, 27.2544041, -74.6959534, 74.7890320
31: -48.6279488, 23.7495174, -49.1020164, 24.0326195, -72.6605682, 72.8515320
32: -48.9266777, 27.2353687, -49.1995163, 27.4923592, -76.4190369, 76.4348831
33: -71.4055862, 43.5921326, -71.8777695, 44.1099625, -115.5155487, 115.4698868
34: -60.6005859, 29.5505180, -60.9779091, 30.0659180, -90.6664886, 90.5284271
35: -56.8148918, 34.3058853, -57.2408409, 34.7528839, -91.5677719, 91.5467224
36: -56.9928322, 33.5350571, -57.3322411, 33.9976501, -90.9904785, 90.8672943
37: -84.6966019, 32.4197121, -85.3449326, 33.1609230, -117.8575287, 117.7646408
38: -68.8271484, 40.6239281, -69.1546936, 40.9938126, -109.8209610, 109.7786255
39: -84.6197968, 40.4210892, -85.0803070, 40.8081818, -125.4279633, 125.5013962
40: -74.9196548, 29.7450790, -75.2646637, 30.0441723, -104.9638290, 105.0097427
41: -54.0271492, 25.5772953, -54.3726425, 26.0081654, -80.0353165, 79.9499359
42: -38.7843552, 29.1340008, -38.9776230, 29.4434128, -68.2277679, 68.1116180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=205, inp2_unstable=207, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=402, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 619
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 976

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 713

## Relational analysis of IS_A1_A2_B2_A1_B2_B2_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B2_B2_A1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.7115718, upper bound: 38.9350457
time: 72.39 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2_B2_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2_B2_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.7125056, upper bound: 38.9865371
time: 79.76 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B2_B2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -52.8780899, 42.8115692, -53.4187393, 43.0612564, -95.9393463, 96.2303009
1: -31.3054581, 35.8716354, -31.6890011, 36.0919952, -67.3974533, 67.5606384
2: -30.0129986, 35.3769989, -30.4425564, 35.6618538, -65.6748505, 65.8195496
3: -33.4851761, 41.2955589, -33.9621201, 41.6553841, -75.1405640, 75.2576752
4: -39.6640739, 38.6839218, -40.0776062, 38.9721603, -78.6362228, 78.7615204
5: -36.4393959, 41.0848236, -36.9089584, 41.4384117, -77.8778076, 77.9937820
6: -55.7771950, 22.3425694, -55.9767838, 22.5091839, -78.2863770, 78.3193512
7: -42.4534073, 39.9323196, -42.9787903, 40.2469025, -82.7003098, 82.9111023
8: -38.8978043, 45.1990585, -39.4351997, 45.5736198, -84.4714203, 84.6342545
9: -33.9258080, 37.3771896, -34.2238426, 37.5384064, -71.4642029, 71.6010284
10: -54.9380875, 52.0861092, -55.2651634, 52.3721352, -107.3102188, 107.3512726
11: -56.2106934, 39.4099884, -56.5807228, 39.7189865, -95.9296799, 95.9907074
12: -58.9065170, 43.6386566, -59.2241592, 44.0734482, -102.9799652, 102.8628159
13: -48.3717651, 49.2684746, -48.7374039, 49.6461945, -98.0179596, 98.0058746
14: -81.0126038, 43.0428314, -81.6289368, 43.3921280, -124.4047318, 124.6717682
15: -40.1642990, 36.2509727, -40.4494133, 36.4037323, -76.5680237, 76.7003784
16: -57.9882393, 40.7523575, -58.3120651, 40.8805618, -98.8688049, 99.0644226
17: -84.9296799, 62.2296219, -85.3008270, 62.5252647, -147.4549408, 147.5304565
18: -48.7393951, 28.7474766, -49.0765800, 29.1194592, -77.8588562, 77.8240509
19: -41.1498413, 19.2260246, -41.4485054, 19.5042877, -60.6541252, 60.6745262
20: -35.2571373, 21.5756531, -35.4476547, 21.8055954, -57.0627289, 57.0233078
21: -48.9521599, 25.1888771, -49.2616119, 25.4613514, -74.4135132, 74.4504852
22: -50.7005730, 29.6756325, -51.0759468, 30.0993404, -80.7999039, 80.7515793
23: -38.8755074, 26.2537155, -39.2392387, 26.5891094, -65.4646149, 65.4929504
24: -44.8998566, 22.5182228, -45.3169289, 22.8463097, -67.7461700, 67.8351517
25: -38.2521591, 30.6254539, -38.6039047, 31.0488911, -69.3010483, 69.2293549
26: -58.7518082, 37.0237198, -59.1933937, 37.6477394, -96.3995514, 96.2171173
27: -49.1127930, 27.0977726, -49.4403992, 27.3441219, -76.4569092, 76.5381699
28: -37.6218987, 28.5017357, -37.9350052, 28.8350601, -66.4569550, 66.4367371
29: -55.1128044, 33.9916992, -55.5528831, 34.3781471, -89.4909515, 89.5445862
30: -47.5079918, 26.9692497, -47.8722305, 27.2544041, -74.7623978, 74.8414764
31: -48.6944847, 23.7655563, -49.1020164, 24.0326195, -72.7271042, 72.8675690
32: -48.9800758, 27.3113403, -49.1995163, 27.4923592, -76.4724274, 76.5108566
33: -71.5018311, 43.7314491, -71.8777695, 44.1099625, -115.6117935, 115.6092148
34: -60.6834869, 29.6750336, -60.9779091, 30.0659180, -90.7494049, 90.6529388
35: -56.9095154, 34.4269409, -57.2408409, 34.7528839, -91.6623917, 91.6677780
36: -57.0749359, 33.6661644, -57.3322411, 33.9976501, -91.0725708, 90.9984055
37: -84.8657608, 32.6515961, -85.3449326, 33.1609230, -118.0266800, 117.9965286
38: -68.8986359, 40.7269135, -69.1546936, 40.9938126, -109.8924408, 109.8816071
39: -84.7279358, 40.5523758, -85.0803070, 40.8081818, -125.5361176, 125.6326828
40: -75.0151825, 29.8647137, -75.2646637, 30.0441723, -105.0593491, 105.1293716
41: -54.1034737, 25.7040634, -54.3726425, 26.0081654, -80.1116409, 80.0767059
42: -38.8060608, 29.1965923, -38.9776230, 29.4434128, -68.2494736, 68.1742096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=205, inp2_unstable=207, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=404, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 596
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 633
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 661
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 645
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 976

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 621

## Relational analysis of IS_A1_A2_B2_A1_B2_B2_A1_A1_B2_A2_A1

### Relational analysis result of IS_A1_A2_B2_A1_B2_B2_A1_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.6221246, upper bound: 38.8612770
time: 83.16 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2_B2_A1_A1_B2_A2_A2

### Relational analysis result of IS_A1_A2_B2_A1_B2_B2_A1_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.7139065, upper bound: 38.9227520
time: 85.14 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B2_B2_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -52.7892685, 42.7863693, -53.4432335, 43.0663834, -95.8556442, 96.2295990
1: -31.2379360, 35.8381157, -31.7045612, 36.0970154, -67.3349533, 67.5426788
2: -29.9821835, 35.3882523, -30.4725246, 35.6680527, -65.6502380, 65.8607788
3: -33.4429970, 41.3163376, -34.0002518, 41.6637878, -75.1067810, 75.3165894
4: -39.6566849, 38.6941872, -40.1079369, 38.9790306, -78.6357117, 78.8021240
5: -36.3996887, 41.1012955, -36.9469452, 41.4473190, -77.8470078, 78.0482407
6: -55.7947807, 22.3137360, -55.9907074, 22.5203323, -78.3151093, 78.3044434
7: -42.3583527, 39.8761292, -43.0056953, 40.2531013, -82.6114502, 82.8818207
8: -38.7942429, 45.1379242, -39.4534836, 45.5848236, -84.3790665, 84.5914001
9: -33.9403305, 37.3559761, -34.2444344, 37.5415955, -71.4819183, 71.6004105
10: -54.9675903, 52.1782303, -55.2795486, 52.4166336, -107.3842163, 107.4577789
11: -56.2526283, 39.4912338, -56.5891571, 39.7608604, -96.0134888, 96.0803909
12: -58.9046822, 43.5661087, -59.2346878, 44.1025620, -103.0072479, 102.8007965
13: -48.5254288, 49.3946571, -48.8035431, 49.6573448, -98.1827698, 98.1981964
14: -80.9221039, 43.0612373, -81.6473694, 43.4294281, -124.3515320, 124.7086029
15: -40.1176605, 36.2486038, -40.4659653, 36.4121399, -76.5298004, 76.7145691
16: -57.9965019, 40.7105865, -58.3315849, 40.8841705, -98.8806763, 99.0421677
17: -84.8453674, 62.2456055, -85.3157196, 62.5448608, -147.3902283, 147.5613098
18: -48.7540741, 28.8221149, -49.0887871, 29.1696167, -77.9236908, 77.9108963
19: -41.1739120, 19.2648087, -41.4571686, 19.5303078, -60.7042198, 60.7219772
20: -35.2755356, 21.6206036, -35.4555511, 21.8300171, -57.1055527, 57.0761566
21: -48.9928970, 25.2583942, -49.2720718, 25.4919624, -74.4848557, 74.5304565
22: -50.6971130, 29.6925163, -51.0872574, 30.1272697, -80.8243866, 80.7797699
23: -38.9281616, 26.3230724, -39.2468834, 26.6258392, -65.5540009, 65.5699539
24: -44.9312401, 22.5717621, -45.3262901, 22.8792896, -67.8105240, 67.8980408
25: -38.2786407, 30.6520252, -38.6131935, 31.0791893, -69.3578339, 69.2652206
26: -58.7542458, 37.0299377, -59.2028732, 37.7032547, -96.4574890, 96.2328110
27: -49.1844215, 27.2094917, -49.4526176, 27.3899899, -76.5744095, 76.6621094
28: -37.6601105, 28.5569153, -37.9422913, 28.8672943, -66.5274048, 66.4992065
29: -55.1094131, 34.0521927, -55.5642891, 34.4133530, -89.5227661, 89.6164780
30: -47.5260620, 27.0123196, -47.8799896, 27.2841759, -74.8102417, 74.8923035
31: -48.7540359, 23.8376579, -49.1155396, 24.0613403, -72.8153687, 72.9531937
32: -49.0056381, 27.2777214, -49.2126694, 27.5051155, -76.5107422, 76.4903870
33: -71.5488129, 43.6550980, -71.9204788, 44.1188393, -115.6676483, 115.5755768
34: -60.6567688, 29.6357803, -60.9900055, 30.0901661, -90.7469254, 90.6257858
35: -56.9288940, 34.3614807, -57.2743149, 34.7617149, -91.6906128, 91.6357880
36: -57.0737343, 33.5607643, -57.3537521, 34.0043945, -91.0781250, 90.9145203
37: -84.7884979, 32.4690819, -85.3652191, 33.1732483, -117.9617462, 117.8342972
38: -68.9101410, 40.6895714, -69.1766891, 41.0092850, -109.9194260, 109.8662567
39: -84.7685852, 40.4656219, -85.1191864, 40.8135071, -125.5820923, 125.5848083
40: -74.9915848, 29.7792664, -75.2803345, 30.0525742, -105.0441513, 105.0596008
41: -54.1004295, 25.6478348, -54.3848419, 26.0280037, -80.1284332, 80.0326691
42: -38.8410988, 29.2158089, -38.9849548, 29.4680023, -68.3090973, 68.2007599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=205, inp2_unstable=207, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=404, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 603
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1491
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 585
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 583
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 713

## Relational analysis of IS_A1_A2_B2_A1_B2_B2_A2_A1_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B2_B2_A2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.7525726, upper bound: 38.9371706
time: 113.54 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2_B2_A2_A1_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2_B2_A2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.6497311, upper bound: 38.8796845
time: 129.32 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B2_B2_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -52.9787025, 42.8502502, -53.4432335, 43.0663834, -96.0450897, 96.2934875
1: -31.3604450, 35.8982315, -31.7045612, 36.0970154, -67.4574585, 67.6027908
2: -30.1105080, 35.4494781, -30.4725246, 35.6680527, -65.7785492, 65.9219971
3: -33.6048241, 41.4055939, -34.0002518, 41.6637878, -75.2686157, 75.4058456
4: -39.7650223, 38.7496109, -40.1079369, 38.9790306, -78.7440491, 78.8575439
5: -36.5589600, 41.1880341, -36.9469452, 41.4473190, -78.0062790, 78.1349792
6: -55.8400192, 22.3852615, -55.9907074, 22.5203323, -78.3603516, 78.3759689
7: -42.5400620, 39.9761581, -43.0056953, 40.2531013, -82.7931671, 82.9818497
8: -38.9640274, 45.2594604, -39.4534836, 45.5848236, -84.5488510, 84.7129364
9: -34.0011749, 37.3962631, -34.2444344, 37.5415955, -71.5427704, 71.6407013
10: -55.0388336, 52.2319336, -55.2795486, 52.4166336, -107.4554672, 107.5114822
11: -56.3174057, 39.5406570, -56.5891571, 39.7608604, -96.0782547, 96.1298141
12: -58.9971085, 43.7359047, -59.2346878, 44.1025620, -103.0996704, 102.9705963
13: -48.5785904, 49.4391556, -48.8035431, 49.6573448, -98.2359314, 98.2426987
14: -81.1462784, 43.1615067, -81.6473694, 43.4294281, -124.5756989, 124.8088760
15: -40.2238159, 36.2906380, -40.4659653, 36.4121399, -76.6359558, 76.7566071
16: -58.0660515, 40.7655106, -58.3315849, 40.8841705, -98.9502258, 99.0970917
17: -84.9928207, 62.3072853, -85.3157196, 62.5448608, -147.5376740, 147.6230011
18: -48.8443108, 28.9058533, -49.0887871, 29.1696167, -78.0139236, 77.9946442
19: -41.2322350, 19.3050632, -41.4571686, 19.5303078, -60.7625389, 60.7622299
20: -35.3162842, 21.6578064, -35.4555511, 21.8300171, -57.1462936, 57.1133575
21: -49.0443573, 25.2834835, -49.2720718, 25.4919624, -74.5363083, 74.5555573
22: -50.7884216, 29.7640781, -51.0872574, 30.1272697, -80.9156952, 80.8513336
23: -38.9795952, 26.3683586, -39.2468834, 26.6258392, -65.6054382, 65.6152420
24: -45.0129662, 22.6232281, -45.3262901, 22.8792896, -67.8922501, 67.9495087
25: -38.3349380, 30.7227097, -38.6131935, 31.0791893, -69.4141235, 69.3359070
26: -58.8609695, 37.2037430, -59.2028732, 37.7032547, -96.5642242, 96.4066162
27: -49.2494202, 27.2387333, -49.4526176, 27.3899899, -76.6394119, 76.6913528
28: -37.7081528, 28.6079350, -37.9422913, 28.8672943, -66.5754471, 66.5502243
29: -55.2235146, 34.1025925, -55.5642891, 34.4133530, -89.6368637, 89.6668854
30: -47.5925293, 27.0647793, -47.8799896, 27.2841759, -74.8767090, 74.9447632
31: -48.8205566, 23.8536873, -49.1155396, 24.0613403, -72.8818970, 72.9692230
32: -49.0590401, 27.3536625, -49.2126694, 27.5051155, -76.5641479, 76.5663300
33: -71.6451111, 43.7943459, -71.9204788, 44.1188393, -115.7639465, 115.7148209
34: -60.7396774, 29.7602692, -60.9900055, 30.0901661, -90.8298416, 90.7502747
35: -57.0235519, 34.4825592, -57.2743149, 34.7617149, -91.7852478, 91.7568665
36: -57.1558418, 33.6918869, -57.3537521, 34.0043945, -91.1602325, 91.0456390
37: -84.9577332, 32.7009506, -85.3652191, 33.1732483, -118.1309738, 118.0661621
38: -68.9816437, 40.7925491, -69.1766891, 41.0092850, -109.9909286, 109.9692383
39: -84.8767242, 40.5969162, -85.1191864, 40.8135071, -125.6902313, 125.7161026
40: -75.0871887, 29.8989410, -75.2803345, 30.0525742, -105.1397629, 105.1792755
41: -54.1767426, 25.7746239, -54.3848419, 26.0280037, -80.2047424, 80.1594543
42: -38.8628159, 29.2783871, -38.9849548, 29.4680023, -68.3308182, 68.2633362

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=205, inp2_unstable=207, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=406, inp2_unstable=407, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 669
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 587
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 653
type: B, layer: 1, pos: 653
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 596
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 652
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 599
type: A, layer: 1, pos: 633
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 599
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 613
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 600
type: A, layer: 1, pos: 600
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 713

## Relational analysis of IS_A1_A2_B2_A1_B2_B2_A2_A1_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A1_B2_B2_A2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.7525726, upper bound: 38.8721842
time: 71.50 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2_B2_A2_A1_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2_B2_A2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.6507314, upper bound: 38.9243095
time: 75.48 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 149.54 seconds
IS_A1_A1_A1_A1_A1_B2_A1_B2_A1_B2_B1, status: Status.VERIFIED, split count: 11, time: 149.54
Output dim: 2, lower bound: -38.6202687, upper bound: 38.9456614
IS_A1_A1_A1_A1_A1_B2_A1_B2_A1_B2_B2, status: Status.VERIFIED, split count: 11, time: 149.54
Output dim: 2, lower bound: -38.6202687, upper bound: 38.9282819
IS_A1_A1_A1_A1_A2_B2_A1_B2_A1_B2_B1, status: Status.VERIFIED, split count: 11, time: 149.54
Output dim: 2, lower bound: -38.6542580, upper bound: 38.9471690
IS_A1_A1_A1_A1_A2_B2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 11, time: 149.54
Output dim: 2, lower bound: -38.6542580, upper bound: 38.9815413
IS_A1_A2_B2_A1_B2_B2_A1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 11, time: 149.54
Output dim: 2, lower bound: -38.7115718, upper bound: 38.9350457
IS_A1_A2_B2_A1_B2_B2_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 11, time: 149.54
Output dim: 2, lower bound: -38.7125056, upper bound: 38.9865371
IS_A1_A2_B2_A1_B2_B2_A1_A1_B2_A2_A1, status: Status.VERIFIED, split count: 11, time: 149.54
Output dim: 2, lower bound: -38.6221246, upper bound: 38.8612770
IS_A1_A2_B2_A1_B2_B2_A1_A1_B2_A2_A2, status: Status.VERIFIED, split count: 11, time: 149.54
Output dim: 2, lower bound: -38.7139065, upper bound: 38.9227520
IS_A1_A2_B2_A1_B2_B2_A2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 11, time: 149.54
Output dim: 2, lower bound: -38.7525726, upper bound: 38.9371706
IS_A1_A2_B2_A1_B2_B2_A2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 11, time: 149.54
Output dim: 2, lower bound: -38.6497311, upper bound: 38.8796845
IS_A1_A2_B2_A1_B2_B2_A2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 11, time: 149.54
Output dim: 2, lower bound: -38.7525726, upper bound: 38.8721842
IS_A1_A2_B2_A1_B2_B2_A2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 11, time: 149.54
Output dim: 2, lower bound: -38.6507314, upper bound: 38.9243095

## BFS IS instance: IS_A1_A1_A1_A1_A2_B2_A1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -52.4859848, 42.6289291, -53.3472519, 43.0464706, -95.5324402, 95.9761810
1: -31.0419083, 35.7150650, -31.6385708, 36.0806770, -67.1225891, 67.3536377
2: -29.7493210, 35.2365570, -30.3933868, 35.6506081, -65.3999329, 65.6299438
3: -33.1810226, 41.1203194, -33.9106979, 41.6436424, -74.8246613, 75.0310211
4: -39.3962860, 38.5279770, -40.0210876, 38.9615822, -78.3578644, 78.5490570
5: -36.1498413, 40.9053574, -36.8622513, 41.4249802, -77.5748138, 77.7676086
6: -55.6746140, 22.1143913, -55.9459877, 22.4531708, -78.1277695, 78.0603790
7: -42.1250153, 39.7448120, -42.9225616, 40.2326508, -82.3576508, 82.6673737
8: -38.5435219, 44.9613495, -39.3678474, 45.5645638, -84.1080856, 84.3291931
9: -33.6986160, 37.2736740, -34.1682892, 37.5185852, -71.2171936, 71.4419632
10: -54.8164368, 51.9697380, -55.2384224, 52.3505630, -107.1669998, 107.2081604
11: -56.0577812, 39.2203789, -56.5384674, 39.6681595, -95.7259369, 95.7588501
12: -58.6956863, 43.2644157, -59.2097359, 44.0037003, -102.6993866, 102.4741516
13: -48.2378082, 49.2095757, -48.7137070, 49.5890770, -97.8268890, 97.9232788
14: -80.7360077, 42.9196472, -81.5888443, 43.3821259, -124.1181335, 124.5084839
15: -39.8710861, 36.1411018, -40.3834801, 36.3581543, -76.2292404, 76.5245819
16: -57.7614861, 40.6511688, -58.2515411, 40.8811722, -98.6426544, 98.9027100
17: -84.6600876, 62.1643448, -85.2565765, 62.5304031, -147.1904907, 147.4208984
18: -48.5908012, 28.5031528, -49.0395050, 29.0656738, -77.6564636, 77.5426483
19: -41.0403862, 19.0824203, -41.4369049, 19.4687805, -60.5091667, 60.5193214
20: -35.1461449, 21.4130630, -35.4298859, 21.7625294, -56.9086761, 56.8429451
21: -48.8322830, 25.0584145, -49.2410393, 25.4250870, -74.2573700, 74.2994537
22: -50.5499725, 29.4833393, -51.0494423, 30.0492306, -80.5991898, 80.5327835
23: -38.7570190, 26.0978813, -39.2255249, 26.5510292, -65.3080444, 65.3233948
24: -44.7661705, 22.3845482, -45.2991905, 22.8176270, -67.5838013, 67.6837387
25: -38.1347733, 30.4370728, -38.5809822, 31.0002594, -69.1350327, 69.0180511
26: -58.5395699, 36.6366348, -59.1702614, 37.5706329, -96.1101990, 95.8068924
27: -48.9895401, 26.9397812, -49.4066315, 27.3013744, -76.2909088, 76.3464050
28: -37.5350761, 28.3407288, -37.9214478, 28.7950974, -66.3301697, 66.2621765
29: -54.9513321, 33.8550453, -55.5288811, 34.3419609, -89.2932816, 89.3839264
30: -47.3673096, 26.8209305, -47.8443146, 27.2223148, -74.5896225, 74.6652451
31: -48.5552292, 23.6248779, -49.0839157, 23.9904461, -72.5456772, 72.7087860
32: -48.8181038, 27.0312538, -49.1761627, 27.4235935, -76.2416992, 76.2074127
33: -71.4425125, 43.5692978, -71.9093628, 44.0880585, -115.5305710, 115.4786606
34: -60.5143356, 29.3972130, -60.9588661, 30.0114822, -90.5258179, 90.3560638
35: -56.8237381, 34.2287331, -57.2535858, 34.7180214, -91.5417633, 91.4823151
36: -56.9135246, 33.2937279, -57.3288002, 33.9102173, -90.8237457, 90.6225204
37: -84.5555801, 32.2649918, -85.3204956, 33.1021080, -117.6576691, 117.5854874
38: -68.6814194, 40.3191566, -69.1302032, 40.8889008, -109.5703201, 109.4493561
39: -84.5686035, 40.3269272, -85.0725403, 40.7646980, -125.3332977, 125.3994598
40: -74.8344574, 29.6556206, -75.2345657, 30.0070705, -104.8415298, 104.8901825
41: -53.9225845, 25.4005833, -54.3466759, 25.9427586, -79.8653412, 79.7472534
42: -38.7307243, 29.0039062, -38.9546661, 29.3969002, -68.1276245, 67.9585724

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=204, inp2_unstable=207, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=401, inp2_unstable=406, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 637
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 669
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 730
type: B, layer: 1, pos: 730
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 587
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 588
type: B, layer: 1, pos: 588
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 567
type: A, layer: 1, pos: 567
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 619
type: B, layer: 1, pos: 668
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 628
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 652
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1491
type: B, layer: 1, pos: 709
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 709
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 585
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 644
type: A, layer: 1, pos: 644
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 613
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 676
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 676
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 661
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 660
type: A, layer: 1, pos: 660
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 583
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1330
type: B, layer: 1, pos: 1330
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 1282

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 621

## Relational analysis of IS_A1_A1_A1_A1_A2_B2_A1_B2_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_A1_A1_A2_B2_A1_B2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.6480457, upper bound: 38.9306106
time: 83.76 seconds

## Relational analysis of IS_A1_A1_A1_A1_A2_B2_A1_B2_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_A1_A1_A2_B2_A1_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.6524323, upper bound: 38.9806370
time: 101.50 seconds

## Summary of splitting at layer (split count: 11)
- Time for IS candidates: 187.82 seconds
IS_A1_A1_A1_A1_A2_B2_A1_B2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 12, time: 187.82
Output dim: 2, lower bound: -38.6480457, upper bound: 38.9306106
IS_A1_A1_A1_A1_A2_B2_A1_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 12, time: 187.82
Output dim: 2, lower bound: -38.6524323, upper bound: 38.9806370
IS_A1_A2_B2_A1_B2_B2_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 11, time: 187.82
Output dim: 2, lower bound: -38.7125056, upper bound: 38.9865371

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 106.61 + 7157.72 = 7264.33 seconds
