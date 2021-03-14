use FeatureCountSets;
impl FeatureCountSets {
pub fn load_trained_score_params() -> FeatureCountSets {
FeatureCountSets {
hairpin_loop_length_counts: [-0.44817275, -0.16366318, -0.2728267, -0.23861857, -0.21493523, -0.2497904, -0.2653634, -0.2137508, -0.24164103, -0.25062013, -0.23935692, -0.21450703, -0.24780494, -0.22692578, -0.2410427, -0.17748758, -0.18543133, -0.16339996, -0.1358676, -0.08753343, -0.13330156, -0.13354513, -0.12181326, -0.06512418, -0.109378956, -0.007831758, -0.03224373, 0.023963327],
bulge_loop_length_counts: [-0.8271259, -0.47292188, -0.40415406, -0.31799933, -0.2386528, -0.18028638, -0.17278264, -0.18019213, -0.17879961, -0.14013498, -0.13974647, -0.1004137, -0.15160014, -0.10746072, -0.095342755, -0.14221792, -0.1018233, -0.11984852, -0.04762957, -0.0867729, -0.040812247, -0.050121617, -0.07130625, -0.055453513, -0.031692438, 0.033553857, 0.009935525, -0.025473697, -0.016466653, -0.050731268],
interior_loop_length_counts: [-0.39792606, -0.4536956, -0.35248005, -0.36283675, -0.3472887, -0.32986787, -0.32341397, -0.26531863, -0.21447013, -0.26022208, -0.22955933, -0.22059138, -0.22300982, -0.24648787, -0.20013614, -0.19353193, -0.17752634, -0.18620592, -0.16689208, -0.21372493, -0.1290433, -0.18420902, -0.14660347, -0.076492965, -0.07902113, -0.058691032, -0.09068603, -0.022647435, -0.06844244],
interior_loop_length_counts_symm: [-0.00299338, -0.04645545, -0.07946505, -0.08320723, -0.03232611, -0.047036305, -0.03069724, 0.0042260084, -0.0013373616, -0.040763047, 0.015405842, 0.0005783026, -0.021688333, -0.007051772, -0.06068398],
interior_loop_length_counts_asymm: [-0.41384032, -0.20987327, -0.1110458, -0.07213754, -0.03247857, -0.044093348, -0.016646944, -0.041089553, -0.010358542, -0.01295368, 0.0041321106, 0.027857563, 0.05218494, -0.02375556, 0.029670436, -0.024672823, -0.013210974, 0.029236909, -0.039895512, -0.009451923, -0.024174972, 0.02768739, 0.016776044, 0.0051595373, 0.039039187, 0.018306585, -0.05754642, 0.023407254],
stack_count_mat: [[[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.19717251], [0.0, 0.0, 0.3033423, 0.0], [0.0, 0.2643429, 0.0, 0.06647976], [0.1533157, 0.0, 0.07667355, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.27838168], [0.0, 0.0, 0.41251004, 0.0], [0.0, 0.3084568, 0.0, 0.037374128], [0.2643429, 0.0, 0.11961262, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.14920607], [0.0, 0.0, 0.27278337, 0.0], [0.0, 0.41251004, 0.0, 0.15656078], [0.3033423, 0.0, 0.24381521, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.13915284], [0.0, 0.0, 0.24381521, 0.0], [0.0, 0.11961262, 0.0, 0.09666738], [0.07667355, 0.0, 0.03763481, 0.0]]], [[[0.0, 0.0, 0.0, 0.18483087], [0.0, 0.0, 0.14920607, 0.0], [0.0, 0.27838168, 0.0, 0.044013806], [0.19717251, 0.0, 0.13915284, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.044013806], [0.0, 0.0, 0.15656078, 0.0], [0.0, 0.037374128, 0.0, 0.020852167], [0.06647976, 0.0, 0.09666738, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]],
terminal_mismatch_count_mat: [[[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[-0.015103126, 0.0020655135, -0.009730031, -0.06606003], [-0.053219102, 0.036477424, -0.09477256, -0.043509297], [-0.071738124, -0.13656932, -0.04781015, -0.05591713], [-0.10596116, -0.035396896, -0.11069891, 0.0035658907]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[-0.012300373, 0.013502019, 0.043495588, -0.13228233], [-0.047607128, -0.0027650634, -0.0765673, -0.08449687], [0.041790187, -0.100714125, -0.015590904, -0.077100396], [-0.16532883, -0.0003239194, -0.021709861, -0.12495369]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[-0.10291159, -0.025720308, -0.059851397, -0.19455218], [0.0044868626, 0.042997528, -0.11996131, -0.0023808326], [-0.048282303, -0.14633077, -0.0428013, -0.06356695], [-0.15517505, -0.027164096, -0.05576739, -0.046391323]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[-0.027495774, -0.020225441, 0.014851521, -0.105951294], [-0.0026757477, 0.034615252, -0.058819965, 0.003496893], [0.030305615, -0.026027557, -0.036363672, 0.00048961566], [-0.056952618, 0.010383273, 0.019501314, 0.028197736]]], [[[-0.022891467, -0.02377076, -0.05324481, -0.12031479], [-0.032482594, 0.032366127, -0.09774243, -0.073678926], [-0.03130929, -0.12683122, -0.031195212, -0.062408917], [-0.13145584, 0.005604813, -0.07716587, -0.0058827167]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[-0.026313761, 0.006331911, -0.011084718, -0.06993743], [-0.0031476368, -0.019658541, -0.030897647, -0.00088689436], [0.05563005, -0.0276343, 0.030567715, -0.10320768], [-0.09677405, -0.031077605, 0.016876766, -0.050176002]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]],
left_dangle_count_mat: [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.10682709, 0.018967785, -0.011295321, -0.04041587]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.22797829, 0.020610048, 0.07003932, -0.17309594], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [-0.061852835, -0.1660931, -0.014600313, -0.09629379], [0.0, 0.0, 0.0, 0.0], [-0.08114876, -0.04411017, -0.02246039, 0.03465495]], [[-0.11717015, -0.041419894, -0.0505981, -0.15340702], [0.0, 0.0, 0.0, 0.0], [-0.08075458, -0.051906575, 0.018988626, -0.022773758], [0.0, 0.0, 0.0, 0.0]]],
right_dangle_count_mat: [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.07195597, 0.048491202, -0.020558871, -0.040161137]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.15439653, 0.1398539, 0.12808442, -0.0798506], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [-0.06813451, 0.032884393, -0.04990337, -0.019684624], [0.0, 0.0, 0.0, 0.0], [-0.045596708, 0.0030462393, 0.022795752, -0.0006222869]], [[-0.09140054, -0.007950777, 0.053983636, -0.03415977], [0.0, 0.0, 0.0, 0.0], [-0.013326221, -0.010473676, 0.053065922, 0.0012416503], [0.0, 0.0, 0.0, 0.0]]],
helix_end_count_mat: [[0.0, 0.0, 0.0, -0.52533454], [0.0, 0.0, -0.48571414, 0.0], [0.0, -0.5238555, 0.0, -0.32554412], [-0.50659156, 0.0, -0.16729428, 0.0]],
base_pair_count_mat: [[0.0, 0.0, 0.0, 0.6043808], [0.0, 0.0, 0.9321699, 0.0], [0.0, 0.9321699, 0.0, 0.25333583], [0.6043808, 0.0, 0.25333583, 0.0]],
interior_loop_length_count_mat_explicit: [[0.09399843, -0.1030162, -0.06253028, -0.05370516], [-0.1030162, -0.04891911, -0.05646083, -0.0006185502], [-0.06253028, -0.05646083, -0.023479337, 0.010114976], [-0.05370516, -0.0006185502, 0.010114976, 0.023879636]],
bulge_loop_0x1_length_counts: [-0.021901729, -0.1166911, -0.106739335, -0.04296564],
interior_loop_1x1_length_count_mat: [[-0.018880945, 0.11866045, 0.0016147036, -0.023642203], [0.11866045, 0.009337677, 0.007641232, 0.061075225], [0.0016147036, 0.007641232, 0.047311384, -0.037183437], [-0.023642203, 0.061075225, -0.037183437, 0.012095636]],
multi_loop_base_count: -0.4954326,
multi_loop_basepairing_count: -0.5887861,
multi_loop_accessible_baseunpairing_count: -0.24757534,
external_loop_accessible_basepairing_count: -0.50300944,
external_loop_accessible_baseunpairing_count: -0.24077591,
basepair_align_count_mat: [[[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.21071053], [0.0, 0.0, -0.10592687, 0.0], [0.0, 0.02327574, 0.0, 0.004732983], [-0.0327242, 0.0, -0.015791224, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, -0.10592687], [0.0, 0.0, 0.2975102, 0.0], [0.0, -0.10575526, 0.0, -0.012954399], [0.025511924, 0.0, 0.022512533, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.02327574], [0.0, 0.0, -0.10575526, 0.0], [0.0, 0.30881178, 0.0, 0.013947843], [-0.06386387, 0.0, -0.004928844, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.004732983], [0.0, 0.0, -0.012954399, 0.0], [0.0, 0.013947843, 0.0, 0.1317428], [-0.022430468, 0.0, -0.025685703, 0.0]]], [[[0.0, 0.0, 0.0, -0.0327242], [0.0, 0.0, 0.025511924, 0.0], [0.0, -0.06386387, 0.0, -0.022430468], [0.20626037, 0.0, 0.012841363, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, -0.015791224], [0.0, 0.0, 0.022512533, 0.0], [0.0, -0.004928844, 0.0, -0.025685703], [0.012841363, 0.0, 0.063100904, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]],
loop_align_count_mat: [[1.0198072, -0.54819614, -0.38240147, -0.37941465], [-0.54819614, 0.7773564, -0.61760205, -0.30253562], [-0.38240147, -0.61760205, 0.89433503, -0.5680983], [-0.37941465, -0.30253562, -0.5680983, 0.8874461]],
opening_gap_count: -3.3824532,
extending_gap_count: -0.26434535,
hairpin_loop_length_counts_cumulative: [-0.44817275, -0.61183596, -0.8846626, -1.1232812, -1.3382164, -1.5880069, -1.8533702, -2.067121, -2.308762, -2.5593822, -2.7987392, -3.0132463, -3.2610512, -3.487977, -3.7290196, -3.9065073, -4.0919385, -4.2553387, -4.3912063, -4.4787397, -4.6120415, -4.7455864, -4.8673997, -4.9325237, -5.0419025, -5.049734, -5.081978, -5.0580144],
bulge_loop_length_counts_cumulative: [-0.8271259, -1.3000478, -1.7042018, -2.022201, -2.2608538, -2.4411402, -2.6139228, -2.794115, -2.9729147, -3.1130497, -3.2527962, -3.35321, -3.50481, -3.6122708, -3.7076135, -3.8498313, -3.9516547, -4.071503, -4.1191325, -4.2059054, -4.2467175, -4.296839, -4.3681455, -4.423599, -4.4552913, -4.421737, -4.411802, -4.4372754, -4.453742, -4.504473],
interior_loop_length_counts_cumulative: [-0.39792606, -0.8516216, -1.2041017, -1.5669384, -1.9142271, -2.244095, -2.5675092, -2.8328278, -3.047298, -3.30752, -3.5370793, -3.7576706, -3.9806805, -4.2271686, -4.4273047, -4.6208367, -4.798363, -4.984569, -5.151461, -5.365186, -5.4942293, -5.678438, -5.825042, -5.9015346, -5.9805555, -6.0392466, -6.1299324, -6.15258, -6.221022],
interior_loop_length_counts_symm_cumulative: [-0.00299338, -0.04944883, -0.12891388, -0.2121211, -0.2444472, -0.29148352, -0.32218075, -0.31795475, -0.3192921, -0.36005515, -0.3446493, -0.344071, -0.36575934, -0.3728111, -0.4334951],
interior_loop_length_counts_asymm_cumulative: [-0.41384032, -0.6237136, -0.7347594, -0.8068969, -0.8393755, -0.88346887, -0.9001158, -0.9412053, -0.95156384, -0.96451753, -0.96038544, -0.9325279, -0.88034296, -0.9040985, -0.8744281, -0.8991009, -0.91231185, -0.88307494, -0.9229705, -0.9324224, -0.9565974, -0.92891, -0.912134, -0.90697443, -0.86793524, -0.8496286, -0.90717506, -0.8837678],
}
}
}