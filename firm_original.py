import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# 1) Define the data for each Task (focusing on 'al_tca' as an example).
#    Make sure each cost/accuracy array has matching lengths!
##############################################################################

# ------------------------- Task 1 -------------------------
accuracy_al_tca_task1 = [
    39.73,
39.085,
40.601,
41.159,
42.167,
42.811,
43.782,
43.778
    # (Truncated one point so it matches 7 cost points)
]
total_cost_al_tca_task1 = [
    1339.615521,
3422.595093,
6260.40336,
8901.633362,
11547.52609,
14938.50965,
18821.95001,
22007.46082
]
total_cost_direct_task1 = [1708.576203,
3417.152406,
5125.728609,
20619.21444,
23939.52444,
25602.92444,
27266.76444,
28764.24444]
accuracy_direct_task1 = [30.11858492,
30.21519614,
30.27802915,
33.444,
35.232,
38.29,
37.715,
39.251]

total_cost_tca_task1 = [1708.576203,
3417.152406,
5125.728609,
20619.21444,
22275.38444,
23939.52444,
25602.92444,
27266.76444,
28764.24444]
accuracy_tca_task1 = [31.43189919,
38.04514914,
33.52819913,
41.43,
45.313,
49.007,
29.882,
34.615,
40.722]

total_cost_no_tca_task1 = [1452.073277,
2722.796719,
4521.147294,
7904.316902,
11680.6324,
16472.19,
23323.67637,
28532.46642
]
accuracy_no_tca_task1 = [
    19.883,
17.648,
18.067,
40.553,
39.835,
40.221,
35.644,
40.228
]

total_cost_scratch_task1 = [1708.576203,
3417.152406,
5125.728609,
20619.21444,
22275.38444,
23939.52444,
25602.92444,
27266.76444,
28764.24444]
accuracy_scratch_task1 = [
    -11.97213311,
    -10.44627321,
    -2.662359811,
    -5.938,
    -1.586,
    14.616,
    37.752,
51.917,
53.534
]



# ------------------------- Task 2 -------------------------
accuracy_al_tca_task2 = [
    7.755, 16.74, 21.832, 32.849, 36.071,
    29.774, 23.683, 25.623, 30.718
]
total_cost_al_tca_task2 = [
    2388465.458, 6276460.129, 10781929.08,
    15535653.53, 20847794.73, 26558065.82,
    33620819.53, 41554763.18, 46749184.82
]
total_cost_direct_task2 = [2115238.928,
42304778.57,
63457167.85,
69803000.94,
69804657.11,
69806321.25,
69807984.65,
69809648.49,
69811145.97]
accuracy_direct_task2 = [12.38831023,
13.97836908,
18.06597262,
30.381,
29.792,
31.441,
33.237,
24.381,
28.192]

total_cost_tca_task2 = [2115238.928,
42304778.57,
63457167.85,
69803000.94,
69804657.11,
69806321.25,
69807984.65,
69809648.49,
69811145.97
]
accuracy_tca_task2 = [24.80924692,
28.56433306,
32.0963275,
41.43,
45.313,
49.007,
29.882,
34.615,
40.722]

total_cost_no_tca_task2 = [1175679.524,
3435447.475,
5946612.298,
14181845.84,
25404632.53,
37016607.47,
49988149.35,
69808805.57
]
accuracy_no_tca_task2 = [
    19.179,
22.935,
25.478,
39.2,
39.494,
40.948,
41.007,
41.315
]

total_cost_scratch_task2 = [2115238.928,
42304778.57,
63457167.85,
69803000.94,
69804657.11,
69806321.25,
69807984.65,
69809648.49,
69811145.97
    ]
accuracy_scratch_task2 = [
    -17.4737545,
-14.75919865,
-15.8448568,
-13.458,
-14.391,
4.795,
17.6,
17.487,
18.292
]

# ------------------------- Task 3 -------------------------
accuracy_al_tca_task3 = [
   10.893, 10.952, 12.527, 16.368, 20.205,
   32.387, 37.937, 41.136, 44.011, 35.034,
   50.068, 47.56
]
total_cost_al_tca_task3 = [
   808.501932,
1377.832639,
2074.708132,
3038.05547,
4323.797036,
6050.430427,
8075.760241,
10178.23106,
13531.1743,
18414.43667,
24338.29947,
29041.54773
]
total_cost_direct_task3 = [7392.415273,
14784.83055,
22177.24582,
22293.54582,
23949.71582,
25613.85582,
27277.25582,
28941.09582,
30438.57582]
accuracy_direct_task3 = [17.61471783,
15.75334057,
15.20389395,
16.72,
21.215,
25.021,
25.962,
26.274,
26.891]

total_cost_tca_task3 = [7392.415273,
14784.83055,
22177.24582,
22293.54582,
23949.71582,
25613.85582,
27277.25582,
28941.09582,
30438.57582]
accuracy_tca_task3 = [19.58127402,
22.66486717,
26.2887236,
28.249,
28.334,
31.678,
32.189,
32.751,
30.84]

total_cost_no_tca_task3 = [998.9509537,
1845.805092,
5469.553954,
7531.809821,
10513.46893,
13019.95624,
16397.9867,
20803.8504,
24252.15009
]
accuracy_no_tca_task3 = [
    10.969,
24.38,
23.307,
22.971,
22.02,
22.312,
22.575,
23.337,
23.985
]

total_cost_scratch_task3 = [7392.415273,
14784.83055,
22177.24582,
22293.54582,
23949.71582,
25613.85582,
27277.25582,
28941.09582,
30438.57582]
accuracy_scratch_task3 = [
    -18.06106872,
-14.91856583,
-10.87272806,
-11.92,
-12.078,
-9.667,
7.071,
24.831,
30.651
]


# Convert them to numpy arrays for convenience
costs1 = np.array(total_cost_al_tca_task1)
accs1  = np.array(accuracy_al_tca_task1)

costs1_direct = np.array(total_cost_direct_task1)
accs1_direct = np.array(accuracy_direct_task1)

costs1_tca = np.array(total_cost_tca_task1)
accs1_tca = np.array(accuracy_tca_task1)

costs1_no_tca = np.array(total_cost_no_tca_task1)
accs1_no_tca = np.array(accuracy_no_tca_task1)

costs1_scratch = np.array(total_cost_scratch_task1)
accs1_scratch = np.array(accuracy_scratch_task1)

################2
costs2 = np.array(total_cost_al_tca_task2)
accs2  = np.array(accuracy_al_tca_task2)

costs2_direct = np.array(total_cost_direct_task2)
accs2_direct = np.array(accuracy_direct_task2)

costs2_tca = np.array(total_cost_tca_task2)
accs2_tca = np.array(accuracy_tca_task2)

costs2_no_tca = np.array(total_cost_no_tca_task2)
accs2_no_tca = np.array(accuracy_no_tca_task2)

costs2_scratch = np.array(total_cost_scratch_task2)
accs2_scratch = np.array(accuracy_scratch_task2)

################3
costs3 = np.array(total_cost_al_tca_task3)
accs3  = np.array(accuracy_al_tca_task3)

costs3_direct = np.array(total_cost_direct_task3)
accs3_direct = np.array(accuracy_direct_task3)

costs3_tca = np.array(total_cost_tca_task3)
accs3_tca = np.array(accuracy_tca_task3)

costs3_no_tca = np.array(total_cost_no_tca_task3)
accs3_no_tca = np.array(accuracy_no_tca_task3)

costs3_scratch = np.array(total_cost_scratch_task3)
accs3_scratch = np.array(accuracy_scratch_task3)

##############################################################################
# 2) Normalize each task’s cost range to [0, 2]
##############################################################################

def normalize_cost_to_range(cost_array, new_min=0.0, new_max=2.0):
    """
    Linearly scale cost_array so that cost_array.min() -> new_min
    and cost_array.max() -> new_max.
    """
    old_min = cost_array.min()
    old_max = cost_array.max()
    return (cost_array - old_min) * (new_max - new_min) / (old_max - old_min) + new_min

norm_costs1 = normalize_cost_to_range(costs1)
norm_costs2 = normalize_cost_to_range(costs2)
norm_costs3 = normalize_cost_to_range(costs3)

norm_costs1_direct = normalize_cost_to_range(costs1_direct)
norm_costs2_direct = normalize_cost_to_range(costs2_direct)
norm_costs3_direct = normalize_cost_to_range(costs3_direct)

norm_costs1_tca = normalize_cost_to_range(costs1_tca)
norm_costs2_tca = normalize_cost_to_range(costs2_tca)
norm_costs3_tca = normalize_cost_to_range(costs3_tca)

norm_costs1_no_tca = normalize_cost_to_range(costs1_no_tca)
norm_costs2_no_tca = normalize_cost_to_range(costs2_no_tca)
norm_costs3_no_tca = normalize_cost_to_range(costs3_no_tca)

norm_costs1_scratch = normalize_cost_to_range(costs1_scratch)
norm_costs2_scratch = normalize_cost_to_range(costs2_scratch)
norm_costs3_scratch = normalize_cost_to_range(costs3_scratch)

##############################################################################
# 3) Interpolate each task’s accuracy onto a common grid in [0,2].
##############################################################################

# We choose a common x-grid with 200 points from 0 to 2
x_common = np.linspace(0, 2, 200)

accs1_interp = np.interp(x_common, norm_costs1, accs1)
accs2_interp = np.interp(x_common, norm_costs2, accs2)
accs3_interp = np.interp(x_common, norm_costs3, accs3)

accs1_direct_interp = np.interp(x_common, norm_costs1_direct, accs1_direct)
accs2_direct_interp = np.interp(x_common, norm_costs2_direct, accs2_direct)
accs3_direct_interp = np.interp(x_common, norm_costs3_direct, accs3_direct)

accs1_tca_interp = np.interp(x_common, norm_costs1_tca, accs1_tca)
accs2_tca_interp = np.interp(x_common, norm_costs2_tca, accs2_tca)
accs3_tca_interp = np.interp(x_common, norm_costs3_tca, accs3_tca)

accs1_no_tca_interp = np.interp(x_common, norm_costs1_no_tca, accs1_no_tca)
accs2_no_tca_interp = np.interp(x_common, norm_costs2_no_tca, accs2_no_tca)
accs3_no_tca_interp = np.interp(x_common, norm_costs3_no_tca, accs3_no_tca)

accs1_scratch_interp = np.interp(x_common, norm_costs1_scratch, accs1_scratch)
accs2_scratch_interp = np.interp(x_common, norm_costs2_scratch, accs2_scratch)
accs3_scratch_interp = np.interp(x_common, norm_costs3_scratch, accs3_scratch)

##############################################################################
# 4) Compute the average accuracy across the three tasks at each grid point.
##############################################################################
accs_mean = (accs1_interp + accs2_interp + accs3_interp) / 3.0

accs_mean_direct = (accs1_direct_interp + accs2_direct_interp + accs3_direct_interp) / 3.0
accs_mean_tca = (accs1_tca_interp + accs2_tca_interp + accs3_tca_interp) / 3.0
accs_mean_no_tca = (accs1_no_tca_interp + accs2_no_tca_interp + accs3_no_tca_interp) / 3.0
accs_mean_scratch = (accs1_scratch_interp + accs2_scratch_interp + accs3_scratch_interp) / 3.0

##############################################################################
# 5) Plot the original data (normalized) and the averaged curve
##############################################################################
plt.figure(figsize=(8,6))

# Plot the raw normalized data for each task:
# plt.plot(norm_costs1, accs1, 'o--', label='Task1 (AL TCA)')
# plt.plot(norm_costs2, accs2, 'o--', label='Task2 (AL TCA)')
# plt.plot(norm_costs3, accs3, 'o--', label='Task3 (AL TCA)')

# Plot the averaged line (in black)
plt.plot(x_common, accs3_interp, 'k-', lw=2, label='Average EMA')

plt.plot(x_common, accs3_direct_interp, 'r-', lw=2, label='Average CL')
plt.plot(x_common, accs3_tca_interp, 'g-', lw=2, label='Average TCA')
plt.plot(x_common, accs3_no_tca_interp, 'b-', lw=2, label='Average EMA W/O TCA')
plt.plot(x_common, accs3_scratch_interp, 'y-', lw=2, label='Average Scratch')

plt.xlabel("Normalized Cost (0 to 2)")
plt.ylabel("Accuracy")
plt.title("workload:readfile")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('firm.pdf')
plt.savefig('firm.png')
