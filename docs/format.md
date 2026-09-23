# vvector snapshot format, version 2

A snapshot is one file of bytes. It travels through Vertica in pieces of up to
8 MB (`vvector.snapshot.chunk`, addressed by `byte_offset`) and is written to a
cache file on every node, which queries map read-only.

- Little-endian only. The engine refuses to compile for a big-endian target.
- The header is 256 bytes. Every section starts on a 64-byte boundary (one
  cache line); padding bytes are 0. The file size is a multiple of 64.
- Readers refuse a file whose version is not 2, whose flags are unknown, or
  whose section offsets differ from what the layout rule below gives. A
  version 1 file (written by milestone M0) is refused with "refresh the index
  to rebuild its snapshot".

## Header (256 bytes)

| Offset | Type | Field | Meaning |
|---:|---|---|---|
| 0 | char[8] | magic | `VVECTOR1` |
| 8 | uint32 | format_version | 2 |
| 12 | uint32 | flags | see below |
| 16 | uint64 | count | number of positions (vectors), tombstoned ones included; at most 4,294,967,295 |
| 24 | uint32 | dims | elements per vector, 1 to 32768 |
| 28 | uint32 | row_stride | floats per stored row: dims rounded up to a multiple of 16 |
| 32 | uint32 | metric | 1 = l2, 2 = cosine, 3 = dot, 4 = l1 |
| 36 | uint32 | reserved0 | 0 |
| 40 | int64 | max_ver | journal watermark of the build |
| 48 | int64 | base_snapshot | snapshot this one was built from; 0 = full build |
| 56 | uint64 | tombstones | number of set bits in the tombstones section |
| 64 | uint64 | checksum | see below |
| 72 | uint64 | total_bytes | file size |
| 80 | uint64 | off_vectors | offset of the vectors section |
| 88 | uint64 | off_ids | offset of the ids section |
| 96 | uint64 | off_id_index | offset of the id_index section, 0 = absent |
| 104 | uint64 | off_tombstones | offset of the tombstones section, 0 = absent |
| 112 | uint64 | off_sq8 | offset of the sq8 section, 0 = absent |
| 120 | uint64 | sq8_bytes | size of the sq8 section |
| 128 | uint64 | off_graph | offset of the graph section, 0 = absent |
| 136 | uint64 | graph_bytes | size of the graph section |
| 144 | uint64[14] | reserved | 0 |

Flags:

| Bit | Name | Meaning |
|---:|---|---|
| 1 | FLAG_HNSW | graph section present (milestone M2) |
| 2 | FLAG_NORMALISED | vectors have unit length; set for cosine indexes and only for them |
| 4 | FLAG_SQ8 | int8 codes section present (milestone M4) |
| 8 | FLAG_ID_INDEX | id_index section present; ids are then in any order (milestone M3) |
| 16 | FLAG_TOMBSTONES | tombstones section present (milestone M3) |

## Sections

In this order, each starting on the next 64-byte boundary after the one before:

1. `vectors`: float32[count x row_stride], row i = position i. Elements
   dims to row_stride - 1 of every row are 0. Vertica FLOAT is 64-bit; the
   snapshot stores 32 bits per element. Cosine rows have unit length (a zero
   vector stays zero).
2. `ids`: int64[count], the id of position i. Strictly ascending when
   FLAG_ID_INDEX is absent (every full build).
3. `id_index` (FLAG_ID_INDEX): uint32[count], the positions sorted by id.
   An id is found by a binary search over `ids[id_index[k]]`.
4. `tombstones` (FLAG_TOMBSTONES): uint64[(count + 63) / 64], one bit per
   position (bit i % 64 of word i / 64); 1 = the position is deleted.
5. `sq8` (FLAG_SQ8): int8 codes and their scale. Defined with milestone M4.
6. `graph` (FLAG_HNSW): the HNSW links. Defined with milestone M2.

Vectors come first so the builder can write every row straight into the
final buffer as it arrives. Rows are sorted by id at the end by moving them
in place, so a build never holds two copies of the vectors.

## Checksum

XOR over all 8-byte words w at word position p (offset / 8) of the file, the
checksum field itself counted as 0, of splitmix64(w + (p + 1) x
0x9E3779B97F4A7C15). Zero words contribute nothing. Because of the XOR, parts
of the file can be summed separately and combined in any order.
vload verifies it, and that the ids are unique and ascending (through the
id_index when present) and that the tombstone count matches; queries check
the header only.
