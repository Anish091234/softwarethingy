"""STL file import/export for LunaRad.

Supports both binary and ASCII STL formats with auto-detection.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .primitives import MeshData


def read_stl(filepath: str | Path) -> MeshData:
    """Read an STL file (auto-detects binary vs ASCII).

    Returns a MeshData with deduplicated vertices.
    """
    filepath = Path(filepath)
    data = filepath.read_bytes()

    if _is_ascii_stl(data):
        return _read_ascii_stl(data)
    else:
        return _read_binary_stl(data)


def write_stl(mesh: MeshData, filepath: str | Path, header: str = "LunaRad export"):
    """Write a binary STL file."""
    filepath = Path(filepath)
    n_faces = mesh.num_faces

    buf = bytearray()
    hdr = header.encode("ascii")[:80].ljust(80, b"\x00")
    buf.extend(hdr)
    buf.extend(struct.pack("<I", n_faces))

    normals = mesh.normals if mesh.normals is not None else mesh._compute_normals()

    for i in range(n_faces):
        n = normals[i]
        buf.extend(struct.pack("<3f", float(n[0]), float(n[1]), float(n[2])))
        for vi in range(3):
            v = mesh.vertices[mesh.faces[i, vi]]
            buf.extend(struct.pack("<3f", float(v[0]), float(v[1]), float(v[2])))
        buf.extend(struct.pack("<H", 0))

    filepath.write_bytes(bytes(buf))


def _is_ascii_stl(data: bytes) -> bool:
    try:
        header = data[:80].decode("ascii", errors="ignore").strip().lower()
        if not header.startswith("solid"):
            return False
        text = data[:2048].decode("ascii", errors="ignore")
        return "facet" in text
    except Exception:
        return False


def _read_binary_stl(data: bytes) -> MeshData:
    num_tris = struct.unpack_from("<I", data, 80)[0]
    offset = 84

    all_verts = np.zeros((num_tris * 3, 3), dtype=np.float64)

    for i in range(num_tris):
        offset += 12  # skip normal
        for v in range(3):
            x, y, z = struct.unpack_from("<3f", data, offset)
            offset += 12
            all_verts[i * 3 + v] = [x, y, z]
        offset += 2  # attribute byte count

    unique_verts, inverse = np.unique(all_verts, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int32)

    return MeshData(vertices=unique_verts, faces=faces)


def _read_ascii_stl(data: bytes) -> MeshData:
    text = data.decode("ascii", errors="ignore")
    lines = [line.strip() for line in text.splitlines()]

    verts_list = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("vertex"):
            parts = lines[i].split()
            verts_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
        i += 1

    if not verts_list:
        raise ValueError("No vertices found in ASCII STL")

    all_verts = np.array(verts_list, dtype=np.float64)
    unique_verts, inverse = np.unique(all_verts, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int32)

    return MeshData(vertices=unique_verts, faces=faces)
