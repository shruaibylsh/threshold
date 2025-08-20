import Rhino.Geometry as rg
import Rhino
import scriptcontext as sc
import math
import System.IO as IO
import System
import System.Threading.Tasks as tasks

SaveFolder = r"C:\Users\shrua\OneDrive\Desktop\threshold project\threshold\data"
RayCount = 2048
MaxDist = 10.0

if Start:

    # Fibonacci Sphere for Ray Directions
    def sphere_directions(n):
        dirs = []
        offset = 2.0 / n
        inc = math.pi * (3.0 - math.sqrt(5.0))
        for i in range(n):
            y = ((i * offset) - 1) + (offset / 2)
            r = math.sqrt(1 - y * y)
            phi = i * inc
            x = math.cos(phi) * r
            z = math.sin(phi) * r
            dirs.append(rg.Vector3d(x, y, z))
        return dirs

    # Preprocess Geometry
    Geo = list(Geo) if isinstance(Geo, (list, tuple)) else [Geo]
    meshes = []
    for ref in Geo:
        rh_obj = sc.doc.Objects.Find(ref)
        if rh_obj:
            geom = rh_obj.Geometry
            if isinstance(geom, rg.Brep):
                mesh_array = rg.Mesh.CreateFromBrep(geom, rg.MeshingParameters.Default)
                if mesh_array:
                    for m in mesh_array:
                        m.RebuildNormals()
                        meshes.append(m)
            elif isinstance(geom, rg.Mesh):
                geom.RebuildNormals()
                meshes.append(geom)

    # Generate directions and unitize
    dirs = sphere_directions(RayCount)
    for d in dirs:
        d.Unitize()

    RayLines = []
    ExportedFiles = []

    # Handle multiple branches of points
    for i, branch in enumerate(Pts.Branches):

        # Convert branch points to Point3d
        real_pts = []
        for ref in branch:
            rh_obj = sc.doc.Objects.Find(ref)
            if rh_obj and hasattr(rh_obj.Geometry, "Location"):
                real_pts.append(rh_obj.Geometry.Location)
            elif isinstance(ref, rg.Point3d):
                real_pts.append(ref)

        Isovists = [None] * len(real_pts)

        # Define function for a single point
        def compute_isovist(idx):
            pt = real_pts[idx]
            dists = []
            for dir in dirs:
                ray = rg.Ray3d(pt, dir)
                dist = MaxDist
                for mesh in meshes:
                    t = rg.Intersect.Intersection.MeshRay(mesh, ray)
                    if t > 0 and t < dist:
                        dist = t
                dists.append(dist)
                # Add debug line
                RayLines.append(rg.Line(pt, dir * dist))
            Isovists[idx] = dists

        # Parallel processing of points
        tasks.Parallel.For(0, len(real_pts), compute_isovist)

        # Save CSV
        if not IO.Directory.Exists(SaveFolder):
            IO.Directory.CreateDirectory(SaveFolder)

        filename = "curve_{:02d}.csv".format(i + 1)
        filepath = IO.Path.Combine(SaveFolder, filename)
        writer = IO.StreamWriter(filepath)

        # Transpose: rows = rays, columns = points
        transposed = zip(*Isovists)
        for ray_row in transposed:
            line = ",".join([str(round(d, 4)) for d in ray_row])
            writer.WriteLine(line)
        writer.Close()

        ExportedFiles.append(filename)
        print("Saved:", filename, "with", len(Isovists), "samples →", filepath)

    # Outputs
    a = ExportedFiles
    b = RayLines

else:
    print("not started")
    a = []
    b = []
