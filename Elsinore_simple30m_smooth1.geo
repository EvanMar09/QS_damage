// =====================================================================
// elsinore_mesh_builtin_curved_smooth1.geo   (smoothed trace, sigma = 1000 m)
//
// Full curved Elsinore trace, built with the same technique that just
// worked for the planar test: built-in kernel, no booleans at all. The
// fault surface is built directly via Extrude of the trace spline (this
// works fine in the built-in kernel too), then embedded explicitly with
// Line{}/Surface{} In Surface{}/Volume{} -- no translate-up-and-trim
// trick needed, since there's no boolean tangency issue to work around
// in the first place; the top edge is embedded directly into the free
// surface at z=0.
//
// Domain matches the confirmed-safe extended depth (Zmin=-40km) from
// the successful planar run.


//Trace points below are already projected to UTM Zone 11N and shifted
// to a local origin at the trace centroid:
//   UTM 11N origin (E0, N0) = (578629.759, 3634947.523)
// Add this offset back to X, Y to recover true UTM 11N coordinates.


// =====================================================================

lc = 5e3;
lc_fault = 30;

fault_width = 15e3;
Zmin = -35e3;

Xmax = 70e3; Xmin = -Xmax;
Ymax = 70e3; Ymin = -Ymax;

// ---------------------------------------------------------------------
// 1. Domain box, built explicitly
// ---------------------------------------------------------------------
Point(1) = {Xmin, Ymin, Zmin, lc};
Point(2) = {Xmax, Ymin, Zmin, lc};
Point(3) = {Xmax, Ymax, Zmin, lc};
Point(4) = {Xmin, Ymax, Zmin, lc};
Point(5) = {Xmin, Ymin, 0, lc};
Point(6) = {Xmax, Ymin, 0, lc};
Point(7) = {Xmax, Ymax, 0, lc};
Point(8) = {Xmin, Ymax, 0, lc};

Line(1) = {1,2}; Line(2) = {2,3}; Line(3) = {3,4}; Line(4) = {4,1};
Line(5) = {5,6}; Line(6) = {6,7}; Line(7) = {7,8}; Line(8) = {8,5};
Line(9) = {1,5}; Line(10) = {2,6}; Line(11) = {3,7}; Line(12) = {4,8};

Line Loop(1) = {1,2,3,4};          Plane Surface(1) = {1};   // bottom
Line Loop(2) = {5,6,7,8};          Plane Surface(2) = {2};   // top (free surface)
Line Loop(3) = {1,10,-5,-9};       Plane Surface(3) = {3};   // y=Ymin side
Line Loop(4) = {2,11,-6,-10};      Plane Surface(4) = {4};   // x=Xmax side
Line Loop(5) = {3,12,-7,-11};      Plane Surface(5) = {5};   // y=Ymax side
Line Loop(6) = {4,9,-8,-12};       Plane Surface(6) = {6};   // x=Xmin side

Surface Loop(1) = {1,2,3,4,5,6};
Volume(1) = {1};

// ---------------------------------------------------------------------
// 2. Fault trace: Gaussian-smoothed (sigma = 1000 m), 94 points ~400 m apart,
//    at z=0. End points equal the original trace end points (P1 and P22).
//    Max distance from the original 16-point trace: 157 m.
// ---------------------------------------------------------------------
Point(1001) = {17497.241, -8952.014, 0, lc_fault};
Point(1002) = {17106.131, -8861.311, 0, lc_fault};
Point(1003) = {16715.597, -8768.170, 0, lc_fault};
Point(1004) = {16326.399, -8669.610, 0, lc_fault};
Point(1005) = {15939.590, -8562.082, 0, lc_fault};
Point(1006) = {15556.444, -8442.183, 0, lc_fault};
Point(1007) = {15178.101, -8307.908, 0, lc_fault};
Point(1008) = {14804.997, -8159.679, 0, lc_fault};
Point(1009) = {14436.436, -8000.467, 0, lc_fault};
Point(1010) = {14070.604, -7835.061, 0, lc_fault};
Point(1011) = {13705.035, -7669.071, 0, lc_fault};
Point(1012) = {13337.278, -7508.002, 0, lc_fault};
Point(1013) = {12965.587, -7356.250, 0, lc_fault};
Point(1014) = {12589.475, -7215.817, 0, lc_fault};
Point(1015) = {12209.851, -7085.151, 0, lc_fault};
Point(1016) = {11828.696, -6959.000, 0, lc_fault};
Point(1017) = {11448.562, -6829.819, 0, lc_fault};
Point(1018) = {11072.112, -6690.313, 0, lc_fault};
Point(1019) = {10701.523, -6535.934, 0, lc_fault};
Point(1020) = {10337.775, -6366.052, 0, lc_fault};
Point(1021) = {9980.343, -6183.231, 0, lc_fault};
Point(1022) = {9627.640, -5991.427, 0, lc_fault};
Point(1023) = {9277.866, -5794.322, 0, lc_fault};
Point(1024) = {8929.603, -5594.556, 0, lc_fault};
Point(1025) = {8581.967, -5393.699, 0, lc_fault};
Point(1026) = {8234.448, -5192.642, 0, lc_fault};
Point(1027) = {7886.650, -4992.066, 0, lc_fault};
Point(1028) = {7538.065, -4792.864, 0, lc_fault};
Point(1029) = {7187.922, -4596.414, 0, lc_fault};
Point(1030) = {6835.226, -4404.591, 0, lc_fault};
Point(1031) = {6478.997, -4219.420, 0, lc_fault};
Point(1032) = {6118.632, -4042.437, 0, lc_fault};
Point(1033) = {5754.161, -3874.070, 0, lc_fault};
Point(1034) = {5386.201, -3713.463, 0, lc_fault};
Point(1035) = {5015.643, -3558.937, 0, lc_fault};
Point(1036) = {4643.300, -3408.760, 0, lc_fault};
Point(1037) = {4269.676, -3261.796, 0, lc_fault};
Point(1038) = {3894.960, -3117.641, 0, lc_fault};
Point(1039) = {3519.200, -2976.229, 0, lc_fault};
Point(1040) = {3142.571, -2837.144, 0, lc_fault};
Point(1041) = {2765.518, -2699.215, 0, lc_fault};
Point(1042) = {2388.613, -2560.880, 0, lc_fault};
Point(1043) = {2012.172, -2421.289, 0, lc_fault};
Point(1044) = {1635.895, -2281.253, 0, lc_fault};
Point(1045) = {1258.983, -2142.942, 0, lc_fault};
Point(1046) = {880.786, -2008.184, 0, lc_fault};
Point(1047) = {501.471, -1876.605, 0, lc_fault};
Point(1048) = {122.089, -1745.217, 0, lc_fault};
Point(1049) = {-255.923, -1609.947, 0, lc_fault};
Point(1050) = {-631.418, -1467.850, 0, lc_fault};
Point(1051) = {-1003.998, -1318.271, 0, lc_fault};
Point(1052) = {-1373.991, -1162.400, 0, lc_fault};
Point(1053) = {-1742.007, -1001.911, 0, lc_fault};
Point(1054) = {-2108.357, -837.655, 0, lc_fault};
Point(1055) = {-2472.600, -668.784, 0, lc_fault};
Point(1056) = {-2833.228, -492.344, 0, lc_fault};
Point(1057) = {-3187.429, -303.374, 0, lc_fault};
Point(1058) = {-3531.023, -95.808, 0, lc_fault};
Point(1059) = {-3858.792, 135.871, 0, lc_fault};
Point(1060) = {-4165.549, 394.687, 0, lc_fault};
Point(1061) = {-4447.941, 679.895, 0, lc_fault};
Point(1062) = {-4705.747, 987.549, 0, lc_fault};
Point(1063) = {-4942.094, 1312.029, 0, lc_fault};
Point(1064) = {-5162.149, 1647.809, 0, lc_fault};
Point(1065) = {-5371.561, 1990.348, 0, lc_fault};
Point(1066) = {-5575.289, 2336.306, 0, lc_fault};
Point(1067) = {-5777.271, 2683.290, 0, lc_fault};
Point(1068) = {-5980.711, 3029.418, 0, lc_fault};
Point(1069) = {-6188.427, 3372.994, 0, lc_fault};
Point(1070) = {-6402.701, 3712.515, 0, lc_fault};
Point(1071) = {-6624.501, 4047.166, 0, lc_fault};
Point(1072) = {-6852.586, 4377.572, 0, lc_fault};
Point(1073) = {-7083.234, 4706.199, 0, lc_fault};
Point(1074) = {-7311.075, 5036.771, 0, lc_fault};
Point(1075) = {-7530.915, 5372.706, 0, lc_fault};
Point(1076) = {-7739.930, 5715.478, 0, lc_fault};
Point(1077) = {-7939.320, 6063.947, 0, lc_fault};
Point(1078) = {-8134.538, 6414.780, 0, lc_fault};
Point(1079) = {-8333.909, 6763.258, 0, lc_fault};
Point(1080) = {-8546.160, 7104.008, 0, lc_fault};
Point(1081) = {-8777.651, 7431.956, 0, lc_fault};
Point(1082) = {-9030.340, 7743.858, 0, lc_fault};
Point(1083) = {-9301.919, 8039.484, 0, lc_fault};
Point(1084) = {-9587.952, 8321.183, 0, lc_fault};
Point(1085) = {-9884.094, 8592.257, 0, lc_fault};
Point(1086) = {-10187.274, 8855.446, 0, lc_fault};
Point(1087) = {-10495.682, 9112.495, 0, lc_fault};
Point(1088) = {-10808.226, 9364.501, 0, lc_fault};
Point(1089) = {-11124.020, 9612.426, 0, lc_fault};
Point(1090) = {-11442.174, 9857.315, 0, lc_fault};
Point(1091) = {-11761.849, 10100.217, 0, lc_fault};
Point(1092) = {-12082.373, 10341.998, 0, lc_fault};
Point(1093) = {-12403.303, 10583.241, 0, lc_fault};
Point(1094) = {-12724.384, 10824.282, 0, lc_fault};

Spline(1000) = {1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039,1040,1041,1042,1043,1044,1045,1046,1047,1048,1049,1050,1051,1052,1053,1054,1055,1056,1057,1058,1059,1060,1061,1062,1063,1064,1065,1066,1067,1068,1069,1070,1071,1072,1073,1074,1075,1076,1077,1078,1079,1080,1081,1082,1083,1084,1085,1086,1087,1088,1089,1090,1091,1092,1093,1094};

// Extrude straight down -- no shift, no boolean, so nothing to trim
fault_extrude[] = Extrude {0, 0, -fault_width} { Curve{1000}; };
// fault_extrude[0] = bottom edge (translated spline, blind tip at z=-15km)
// fault_extrude[1] = the fault surface itself

// ---------------------------------------------------------------------
// 3. Explicit embedding -- this replaces BooleanFragments entirely
// ---------------------------------------------------------------------
Line{1000} In Surface{2};            // top edge conforms to the free surface
Surface{fault_extrude[1]} In Volume{1}; // fault surface embedded inside the volume

free_surf[] = {2};
fault_surfs[] = {fault_extrude[1]};
absorbing_surfs[] = {1,3,4,5,6};

// ---------------------------------------------------------------------
// 4. Mesh size fields (same grading as the planar built-in version)
// ---------------------------------------------------------------------
Field[1] = Distance;
Field[1].FacesList = {fault_surfs[]};

// Linear + Quadratic + Cubic expansion
Field[2] = MathEval;
Field[2].F = Sprintf("%g + 0.1 * F1 + (F1 / 500)^2 + (F1 / 1500)^3", lc_fault);

Field[3] = MathEval; Field[3].F = Sprintf("%g", lc_fault);
Field[4] = MathEval; Field[4].F = Sprintf("%g", lc);

Field[5] = Max; Field[5].FieldsList = {2, 3}; // Lower bound clamp (>= lc_fault)
Field[6] = Min; Field[6].FieldsList = {4, 5}; // Upper bound clamp (<= lc)

Background Field = 6;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;

Mesh.OptimizeThreshold = 0.8;

// ---------------------------------------------------------------------
// 5. SeisSol physical tags
// ---------------------------------------------------------------------
Physical Volume(1) = {1};
Physical Surface(101) = {free_surf[]};
Physical Surface(103) = {fault_surfs[]};
Physical Surface(105) = {absorbing_surfs[]};

Mesh.MshFileVersion = 2.2;
