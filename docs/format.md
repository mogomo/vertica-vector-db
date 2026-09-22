# vvector snapshot format, version 1

A snapshot is one file of bytes. It travels through Vertica in pieces of up to
8 MB (`vvector.snapshot.chunk`, addressed by `byte_offset`) and is written to a
cache file on every node, which queries map read-only.

- Little-endian only. The engine refuses to compile for a big-endian target.
- Every section starts on an 8-byte boundary; padding bytes are 0.
- The file size is a multiple of 8.

## Header (128 bytes)

| Offset | Type | Field | Meaning |
|---:|---|---|---|
| 0 | char[8] | magic | `VVECTOR1` |
| 8 | uint32 | format_version | 1 |
| 12 | uint32 | flags | 1 = HNSW graph section present (not written yet) |
| 16 | uint64 | count | number of vectors |
| 24 | uint32 | dims | elements per vector, at least 1 |
| 28 | uint32 | metric | 1 = l2, 2 = cosine, 3 = dot |
| 32 | int64 | max_ver | journal watermark of the build |
| 40 | uint64 | checksum | see below |
| 48 | uint64 | total_bytes | file size |
| 56 | uint64 | off_ids | offset of the ids section |
| 64 | uint64 | off_vectors | offset of the vectors section |
| 72 | uint64 | off_graph | offset of the HNSW section, 0 = absent |
| 80 | uint64 | graph_bytes | size of the HNSW section |
| 88 | uint64[5] | reserved | 0 |

The section offsets must be exactly what the layout rule below gives for
count, dims, flags and graph_bytes; a reader refuses anything else.

## Sections

1. `ids`: int64[count], strictly ascending (unique).
2. `vectors`: float32[count x dims], row-major; row i belongs to ids[i].
   Vertica FLOAT is 64-bit; the snapshot stores 32 bits per element.
3. `graph` (only with flag 1): the HNSW links. Defined with milestone M2.

## Checksum

XOR over all 8-byte words w at word position p (offset / 8) of the file, the
checksum field itself counted as 0, of splitmix64(w + (p + 1) x
0x9E3779B97F4A7C15). Zero words contribute nothing. Because of the XOR, parts
of the file can be summed separately and combined in any order.
vload verifies it; queries do not (they check the header only).
