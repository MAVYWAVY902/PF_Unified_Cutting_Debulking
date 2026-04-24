// Volumetric torus for ICRA 2027 capsule task.
// Ring plane = XZ (axis along Y, parallel to gravity so the ring lays flat).
// Center (0.5, 0.25, 0.5) matches cuboid workspace center.

SetFactory("OpenCASCADE");

R  = 0.35;   // major radius (center-of-torus to center-of-tube)
r  = 0.12;   // minor radius (tube radius). ~2.4x knife_radius=0.05.
cx = 0.5;
cy = 0.25;
cz = 0.5;

// gmsh's built-in Torus is axis along Z. Rotate +90deg about X axis so axis is along Y.
Torus(1) = {cx, cy, cz, R, r};
Rotate {{1, 0, 0}, {cx, cy, cz}, Pi/2} { Volume{1}; }

// Mesh controls
Mesh.Algorithm   = 6;   // 2D: Frontal-Delaunay
Mesh.Algorithm3D = 10;  // 3D: HXT (parallel, fast)
Mesh.CharacteristicLengthFromCurvature = 0;
Mesh.CharacteristicLengthExtendFromBoundary = 1;
Mesh.CharacteristicLengthMin = 0.020;
Mesh.CharacteristicLengthMax = 0.028;
Mesh.ElementOrder = 1;  // linear tets

// Output format: legacy .msh 2.2 ASCII (meshio reads this cleanly; load_tet_mesh expects it)
Mesh.MshFileVersion = 2.2;
Mesh.Binary = 0;

Mesh 3;

Save "torus.msh";
