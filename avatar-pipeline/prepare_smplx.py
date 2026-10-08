"""Put the SMPL-X / SMPL files you downloaded where the avatar pipeline expects them.

The body models are licensed by the Max Planck Institute for Intelligent Systems and
cannot be redistributed, so each user downloads them (see README.md). This script

  * copies SMPLX_NEUTRAL.npz to body_models/smplx/ (HAHA) and to
    third_party/mmhuman3d/data/body_models/smplx/ (PyMAF-X);
  * writes body_models/smplx/smplx-10.obj: the UV layout of the SMPL-X Unity add-on
    mesh (smplx-neutral.fbx), read directly from the FBX. The fitted texture uses
    this layout so it fits the Unity avatar, which differs from the UVs in the NPZ;
  * writes third_party/HAHA/metadata/smplx_texture_f_alb_512.png, the add-on's
    female albedo at 512 x 512, which initialises the texture;
  * optionally copies the SMPL neutral model to
    third_party/mmhuman3d/data/body_models/smpl/SMPL_NEUTRAL.pkl (PyMAF-X).

Run it after setup.sh.
"""
import argparse
import os
import shutil
import struct
import sys
import zlib

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))


# --- minimal binary FBX reader (geometry only) ---------------------------------------

_ARRAYS = {b"f": "<f4", b"d": "<f8", b"l": "<i8", b"i": "<i4", b"b": "<u1"}
_SCALARS = {b"Y": ("<h", 2), b"C": ("<?", 1), b"I": ("<i", 4), b"F": ("<f", 4), b"D": ("<d", 8), b"L": ("<q", 8)}


def read_fbx(path):
    """Node tree of a binary FBX file as nested (name, properties, children) tuples."""
    data = open(path, "rb").read()
    if data[:21] != b"Kaydara FBX Binary  \x00":
        raise ValueError(f"{path} is not a binary FBX file")
    version = struct.unpack_from("<I", data, 23)[0]
    head = struct.Struct("<QQQ" if version >= 7500 else "<III")

    def read_node(pos):
        end, num_props, _ = head.unpack_from(data, pos)
        pos += head.size
        if end == 0:
            return None, pos
        name = data[pos + 1:pos + 1 + data[pos]].decode("ascii", "replace")
        pos += 1 + data[pos]
        props = []
        for _ in range(num_props):
            kind = data[pos:pos + 1]
            pos += 1
            if kind in _SCALARS:
                fmt, size = _SCALARS[kind]
                props.append(struct.unpack_from(fmt, data, pos)[0])
                pos += size
            elif kind in _ARRAYS:
                length, encoding, size = struct.unpack_from("<III", data, pos)
                raw = data[pos + 12:pos + 12 + size]
                pos += 12 + size
                props.append(np.frombuffer(zlib.decompress(raw) if encoding == 1 else raw,
                                           dtype=_ARRAYS[kind], count=length))
            elif kind in (b"S", b"R"):
                size = struct.unpack_from("<I", data, pos)[0]
                props.append(data[pos + 4:pos + 4 + size])
                pos += 4 + size
            else:
                raise ValueError(f"unknown FBX property type {kind!r}")
        children = []
        while pos < end:
            node, pos = read_node(pos)
            if node is None:
                break
            children.append(node)
        return (name, props, children), end

    nodes, pos = [], 27
    while pos < len(data):
        node, pos = read_node(pos)
        if node is None:
            break
        nodes.append(node)
    return nodes


def _child(node, name):
    return next((c for c in node[2] if c[0] == name), None)


def fbx_mesh(path):
    """Vertices, triangles and per-corner UVs of the first polygon mesh in an FBX file."""
    objects = next(n for n in read_fbx(path) if n[0] == "Objects")
    for geometry in (n for n in objects[2] if n[0] == "Geometry"):
        polygons = _child(geometry, "PolygonVertexIndex")
        if polygons is None:
            continue  # blend shape
        polygons = polygons[1][0].astype(np.int64)
        if not np.all(np.diff(np.concatenate([[-1], np.where(polygons < 0)[0]])) == 3):
            raise ValueError("expected a triangle mesh")
        faces = np.where(polygons < 0, -polygons - 1, polygons).reshape(-1, 3)
        vertices = _child(geometry, "Vertices")[1][0].reshape(-1, 3)
        layer = _child(geometry, "LayerElementUV")
        if _child(layer, "MappingInformationType")[1][0] != b"ByPolygonVertex" or \
                _child(layer, "ReferenceInformationType")[1][0] != b"IndexToDirect":
            raise ValueError("unexpected UV layout in the FBX")
        uv = _child(layer, "UV")[1][0].reshape(-1, 2)
        uv_faces = _child(layer, "UVIndex")[1][0].astype(np.int64).reshape(-1, 3)
        return vertices, faces, uv, uv_faces
    raise ValueError(f"no mesh in {path}")


def write_uv_obj(path, vertices, faces, uv, uv_faces):
    with open(path, "w") as f:
        f.write("# SMPL-X Unity add-on mesh (smplx-neutral.fbx): UV layout for the avatar texture\n")
        f.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in vertices)
        f.writelines(f"vt {u:.6f} {v:.6f}\n" for u, v in uv)
        f.writelines(f"f {a + 1}/{ta + 1} {b + 1}/{tb + 1} {c + 1}/{tc + 1}\n"
                     for (a, b, c), (ta, tb, tc) in zip(faces, uv_faces))


# --- main ------------------------------------------------------------------------------

def find_file(root, name):
    for dirpath, _, files in os.walk(root):
        if name in files:
            return os.path.join(dirpath, name)
    raise FileNotFoundError(f"{name} not found under {root}")


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"  {dst}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smplx_model", required=True,
                        help="SMPLX_NEUTRAL.npz from the SMPL-X download (v1.0 or v1.1)")
    parser.add_argument("--unity_project", required=True,
                        help="unzipped SMPL-X Unity project (e.g. SMPLX_UnityProject_20241205)")
    parser.add_argument("--smpl_model", default=None,
                        help="SMPL neutral model basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl (PyMAF-X)")
    args = parser.parse_args()

    third_party = os.path.join(ROOT, "third_party")
    if not os.path.isdir(os.path.join(third_party, "HAHA")):
        sys.exit("third_party/ is missing: run ./setup.sh first")

    # The model must have the Unity avatar's mesh: the same triangles and the template of
    # SMPL-X v1.0 / v1.1 (other variants, e.g. without the head bun, have a different head)
    fbx = find_file(args.unity_project, "smplx-neutral.fbx")
    vertices, faces, uv, uv_faces = fbx_mesh(fbx)
    model = np.load(args.smplx_model, allow_pickle=True)
    if not np.array_equal(faces, model["f"]):
        sys.exit(f"the triangles of {fbx} do not match the SMPL-X model")
    if np.abs(vertices - model["v_template"]).max() > 1e-4:
        sys.exit(f"the template of {args.smplx_model} differs from the Unity avatar: use SMPL-X v1.0 or v1.1")

    print("SMPL-X model:")
    copy(args.smplx_model, os.path.join(ROOT, "body_models", "smplx", "SMPLX_NEUTRAL.npz"))
    copy(args.smplx_model, os.path.join(third_party, "mmhuman3d", "data", "body_models", "smplx", "SMPLX_NEUTRAL.npz"))

    print("Unity add-on UV layout and initial texture:")
    obj = os.path.join(ROOT, "body_models", "smplx", "smplx-10.obj")
    write_uv_obj(obj, vertices, faces, uv, uv_faces)
    print(f"  {obj} ({len(uv)} UVs, {len(faces)} triangles)")

    albedo = Image.open(find_file(args.unity_project, "smplx_texture_f_alb.png")).convert("RGB")
    init = os.path.join(third_party, "HAHA", "metadata", "smplx_texture_f_alb_512.png")
    os.makedirs(os.path.dirname(init), exist_ok=True)
    albedo.resize((512, 512), Image.LANCZOS).save(init)
    print(f"  {init}")

    if args.smpl_model:
        print("SMPL model:")
        copy(args.smpl_model, os.path.join(third_party, "mmhuman3d", "data", "body_models", "smpl", "SMPL_NEUTRAL.pkl"))
    print("done")


if __name__ == "__main__":
    main()
