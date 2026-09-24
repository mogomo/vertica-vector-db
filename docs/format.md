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
| 8 | FLAG_ID_INDEX | id_index section present; ids are then in any order (incremental builds, milestone M3) |
| 16 | FLAG_TOMBSTONES | tombstones section present (incremental builds, milestone M3) |

## Sections

In this order, each starting on the next 64-byte boundary after the one before:

1. `vectors`: float32[count x row_stride], row i = position i. Elements
   dims to row_stride - 1 of every row are 0. Vertica FLOAT is 64-bit; the
   snapshot stores 32 bits per element. Cosine rows have unit length (a zero
   vector stays zero).
2. `ids`: int64[count], the id of position i. Strictly ascending when
   FLAG_ID_INDEX is absent (every full build, and an incremental build that
   only appended ids larger than all before). With FLAG_ID_INDEX in any
   order, and an id can be at several positions: an incremental build
   appends a changed vector and tombstones its old position. At most one of
   them is live, and it is the highest.
3. `id_index` (FLAG_ID_INDEX): uint32[count], every position once, sorted
   by id and, for one id, by position from high to low. An id is found by a
   binary search over `ids[id_index[k]]`: its first entry is the only one
   that can be live.
4. `tombstones` (FLAG_TOMBSTONES): uint64[(count + 63) / 64], one bit per
   position (bit i % 64 of word i / 64); 1 = the position is deleted. Bits
   from count on are 0. A tombstoned position keeps its row, its id and its
   links in the graph: searches pass through it but never return it.
5. `sq8` (FLAG_SQ8): one byte per element and the scale that maps it back
   to a float, see below.
6. `graph` (FLAG_HNSW): the HNSW links, see below.

Vectors come first so the builder can write every row straight into the
final buffer as it arrives. Rows are sorted by id at the end by moving them
in place, so a build never holds two copies of the vectors.

## sq8 section (FLAG_SQ8)

Scalar quantisation (milestone M4): every element of every row is also
stored as one byte, a code from 0 to 255. One range holds for the whole
index: element x has the code

    code = min(255, max(0, floor((x - offset) / scale + 0.5)))

computed in float32, and the code stands for `offset + scale x code`. The
`vectors` section stays: searches rank candidates by their codes, then
compute the exact scores of the best candidates from the float rows. Offsets
below are from the start of the sq8 section; every part starts on a 64-byte
boundary, padding bytes are 0.

sq8 header (64 bytes):

| Offset | Type | Field | Meaning |
|---:|---|---|---|
| 0 | float32 | scale | (hi - lo) / 255, or 1 when hi = lo |
| 4 | float32 | offset | lo |
| 8 | uint32 | code_stride | bytes per code row: row_stride |
| 12 | uint32 | sample | elements the range was trained on |
| 16 | uint64 | count | positions, equal to the snapshot's count |
| 24 | uint64[5] | reserved | 0 |

Parts, in this order:

1. `codes`: uint8[count x code_stride], row i = position i. Codes of the
   elements dims to code_stride - 1 are 0 (not the code of 0.0), so the
   padding adds nothing to any sum below.
2. `sums`: uint32[count], the sum of the codes of each row.

Training (a full build): lo and hi are the 0.001 and 0.999 quantiles of a
sample of all elements of evenly spaced rows (in position order), about
100,000 elements; values outside [lo, hi] get code 0 or 255. Cosine rows are
normalised before they are coded. An incremental build keeps the base's
scale and offset and codes the appended rows with them; a full build trains
anew.

What a search computes from the codes (a = row codes, b = query codes, both
coded with the same range, sums over the dims elements, all in uint32, so
the result is exact on every CPU):

| Metric | Integer sum | Key (smaller is closer) |
|---|---|---|
| l2 | sum (a - b)^2 | scale^2 x sum |
| l1 | sum abs(a - b) | scale x sum |
| dot, cosine | sum a x b | -(scale^2 x sum + scale x offset x (sums[i] + sum of b) + dims x offset^2) |

The key is computed in double from the exact integers, in this order, and
rounded to float32. It is the approximate value of the metric's built-in
(for l2 its square, for dot and cosine its negative), which is what a search
reports without rescoring. Every sum fits: 255^2 x 32768 < 2^32.

## Graph section (FLAG_HNSW)

The links of the HNSW graph (Malkov and Yashunin, 2018). The vectors are not
repeated: the graph refers to positions, and a search reads the rows of the
`vectors` section. Offsets below are from the start of the graph section;
every part starts on a 64-byte boundary, padding bytes are 0.

Graph header (64 bytes):

| Offset | Type | Field | Meaning |
|---:|---|---|---|
| 0 | uint32 | m | links per node on the levels above 0, 2 to 256 |
| 4 | uint32 | m0 | links per node on level 0: 2 x m |
| 8 | uint32 | ef_construction | candidate list size of the build (information only) |
| 12 | uint32 | max_level | level of the entry point, at most 32 |
| 16 | uint32 | entry_point | position where every search starts |
| 20 | uint32 | reserved0 | 0 |
| 24 | uint64 | count | positions, equal to the snapshot's count |
| 32 | uint64 | level_seed | seed of the level function |
| 40 | uint64 | upper_blocks | number of blocks in `upper`: the sum of all levels |
| 48 | uint64[2] | reserved | 0 |

Parts, in this order:

1. `levels`: uint8[count], the top level of every position.
2. `level0`: count blocks of (m0 + 1) uint32, one per position: the number
   of links n, then n neighbour positions (the rest of the block is 0).
3. `upper_index`: uint32[count], the index of the first `upper` block of a
   position, 0xFFFFFFFF when its level is 0.
4. `upper`: for every position of level L >= 1, in position order, L blocks
   of (m + 1) uint32 for the levels 1 to L: n, then n neighbour positions.

The level of a vector is floor(-ln(u) / ln(m)) with u in (0, 1] taken from
a hash of (level_seed, id), capped at 32: the same id always gets the same
level. The section size follows from count, m and upper_blocks; readers
refuse a section whose size differs. vload also checks every link (in
range, not to itself, not to a node below that level, lists not longer than
m or m0) and that the levels, upper_index and upper_blocks agree.

## Checksum

XOR over all 8-byte words w at word position p (offset / 8) of the file, the
checksum field itself counted as 0, of splitmix64(w + (p + 1) x
0x9E3779B97F4A7C15). Zero words contribute nothing. Because of the XOR, parts
of the file can be summed separately and combined in any order.
vload verifies it, the ids (without id_index: unique and ascending; with it:
every position listed once, sorted as above, no id live at two positions),
that the tombstone count matches the bits, and the graph links; queries check
the headers only.

## Incremental builds

`base_snapshot` in the header names the snapshot an incremental build
started from (0 = a full build). Such a build copies the sections of the
base, then:

- appends the new and the changed vectors (in id order) as positions
  `count_base` to `count - 1`, with their ids;
- sets the tombstone bit of every deleted or changed id's old position;
- writes an id_index when an appended id is not larger than every id
  before it, or when the base had one;
- extends the sq8 section: the base's codes and sums are copied, the
  appended rows are coded with the base's scale and offset (the header is
  copied unchanged, only count changes);
- extends the graph: `levels`, `level0` and `upper_index` of the base are
  copied to the same positions, the base's `upper` blocks come first in
  `upper` (so a base position keeps its block index), the new positions get
  their levels and blocks after them, and the new nodes are inserted with
  the insertion code of the full build. New nodes are never linked to
  tombstoned ones; a base node's list can change when a new node links back
  to it. The entry point and max_level can change.

A full rebuild writes neither tombstones nor an id_index.
