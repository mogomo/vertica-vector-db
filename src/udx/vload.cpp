// vload: writes the snapshot cache file on every node and makes it active.
//   vvector_admin.vload(byte_offset, chunk, base_snapshot USING PARAMETERS index_name='docs', snapshot_id=7) OVER(PARTITION NODES)
// Output (node_name, snapshot_id, bytes, status). Idempotent: run it again any time.
// A piece is a whole 8 MB chunk at its offset (base_snapshot NULL), or a row of a patch (milestone
// M7): up to 8 MB of packed runs of changed bytes (delta.h pack_runs, each written at its own
// offset); base_snapshot names the snapshot the patch was made on, and the file starts as a copy of
// that snapshot's file in the node's cache (a reflink where the file system has it). Every piece of
// one load carries the same base_snapshot.
// A large snapshot is loaded in passes (vvector.load_on_nodes: a broadcast join holds its inner in
// memory): parameters part (letters and digits naming the load), pass and passes; pass 1 creates a
// partial file, every pass writes its pieces into it, the last one verifies and activates it
// (status 'loaded'), the others return status 'partial'.
// Thin adapter around src/engine/cache.h.
#include "udx_common.h"

#include <algorithm>
#include <sys/stat.h>
#include "../engine/delta.h"

using namespace Vertica;
using namespace vvector_udx;

static const char *const FN = "vload";
// A patch starts from a reflink of the base when its runs are few for the base's size: a write into
// a reflink unshares a copy-on-write extent, 3 to 8 ms each on the xfs disks measured, while a plain
// copy of the base by read and write costs about 10 ms per MB on a 100 MB/s disk (copy_file_range
// would be a reflink again on xfs: cache.h). So a reflink up to one run per 4 MB of the base, at
// least PATCH_REFLINK_RUNS: measured on the 4-node cluster, an HNSW patch of 1000 adds and 500
// deletes loads in 35 s from a copy of a 6.7 GB base and 62 s from a reflink, and in 161 s from a
// reflink of a 66.7 GB base against 704 s from a copy.
static const std::uint64_t PATCH_REFLINK_RUNS = 64;
static bool reflink_for(std::uint64_t runs, std::uint64_t base_bytes)
{
    return runs <= std::max<std::uint64_t>(PATCH_REFLINK_RUNS, base_bytes >> 22);
}

class VLoad : public TransformFunction
{
    virtual void processPartition(ServerInterface &srvInterface,
                                  PartitionReader &inputReader,
                                  PartitionWriter &outputWriter)
    {
        const std::string node = srvInterface.getCurrentNodeName();
        try {
            ParamReader params = srvInterface.getParamReader();
            const std::string name = read_index_name(srvInterface);
            if (!params.containsParameter("snapshot_id")) fail("parameter snapshot_id is required");
            const vint snapshot_id = params.getIntRef("snapshot_id");

            const bool in_parts = params.containsParameter("part");
            vint pass = 1, passes = 1;
            if (in_parts) {
                if (!params.containsParameter("pass") || !params.containsParameter("passes"))
                    fail("parameter part needs pass and passes");
                pass = params.getIntRef("pass");
                passes = params.getIntRef("passes");
                if (passes < 1 || pass < 1 || pass > passes) fail("pass must be 1 to passes");
            }

            vvector::CacheWriter writer;
            // The base comes with the first piece; every piece must name the same one. A patch of a
            // few runs starts from a reflink of the base (free); one of many runs from a copy: every
            // write into a reflink unshares an extent (3 ms each on xfs, cache.h).
            const vint base = inputReader.isNull(2) ? 0 : inputReader.getIntRef(2);
            if (base < 0) fail("index '" + name + "': base_snapshot must be a snapshot id");
            if (base == snapshot_id) fail("index '" + name + "': a snapshot cannot be its own base");
            bool reflink = true;
            if (base != 0) {
                if (inputReader.getStringRef(1).isNull()) fail("index '" + name + "': NULL chunk");
                const VString &first = inputReader.getStringRef(1);
                struct stat base_st;
                const std::uint64_t base_bytes =
                    stat(vvector::snapshot_path(resolve_cache_dir(srvInterface), name, base).c_str(), &base_st) == 0 ? static_cast<std::uint64_t>(base_st.st_size) : 0;
                reflink = reflink_for(vvector::patch_runs_of(reinterpret_cast<const std::uint8_t *>(first.data()), first.length()), base_bytes);
            }
            if (in_parts)
                writer.begin_part(resolve_cache_dir(srvInterface), name, snapshot_id,
                                  params.getStringRef("part").str(), pass > 1, base, reflink);
            else
                writer.begin(resolve_cache_dir(srvInterface), name, snapshot_id, base, reflink);
            do {
                if (inputReader.isNull(0) || inputReader.getStringRef(1).isNull())
                    fail("index '" + name + "': NULL byte_offset or chunk");
                if ((inputReader.isNull(2) ? 0 : inputReader.getIntRef(2)) != base)
                    fail("index '" + name + "': the pieces of snapshot " + std::to_string(snapshot_id) + " name different base snapshots");
                const VString &chunk = inputReader.getStringRef(1);
                if (base == 0) {
                    writer.write_at(inputReader.getIntRef(0), chunk.data(), chunk.length());
                } else {
                    // A patch row: runs packed by vbuild (delta.h), each written at its own offset.
                    vvector::unpack_runs(reinterpret_cast<const std::uint8_t *>(chunk.data()), chunk.length(),
                                         [&](std::uint64_t off, const std::uint8_t *data, std::uint64_t n) {
                                             writer.write_at(static_cast<std::int64_t>(off), reinterpret_cast<const char *>(data), n);
                                         });
                }
                if (isCanceled()) return;
            } while (inputReader.next());
            std::uint64_t bytes = 0;
            if (pass < passes)
                writer.keep();
            else
                bytes = writer.commit();

            outputWriter.getStringRef(0).copy(node);
            outputWriter.setInt(1, snapshot_id);
            outputWriter.setInt(2, (vint)bytes);
            outputWriter.getStringRef(3).copy(pass < passes ? "partial" : "loaded");
            outputWriter.next();
        } catch (std::exception &e) {
            vt_report_error(0, "%s: on %s: %s", FN, node.c_str(), e.what());
        }
    }
};

class VLoadFactory : public TransformFunctionFactory
{
    virtual void getPrototype(ServerInterface &srvInterface, ColumnTypes &argTypes, ColumnTypes &returnType)
    {
        argTypes.addInt();
        argTypes.addLongVarbinary();
        argTypes.addInt();                 // base_snapshot
        returnType.addVarchar();
        returnType.addInt();
        returnType.addInt();
        returnType.addVarchar();
    }

    virtual void getReturnType(ServerInterface &srvInterface, const SizedColumnTypes &inputTypes,
                               SizedColumnTypes &outputTypes)
    {
        outputTypes.addVarchar(128, "node_name");
        outputTypes.addInt("snapshot_id");
        outputTypes.addInt("bytes");
        outputTypes.addVarchar(32, "status");
    }

    virtual void getParameterType(ServerInterface &srvInterface, SizedColumnTypes &parameterTypes)
    {
        add_common_parameters(parameterTypes);
        parameterTypes.addInt("snapshot_id");
        parameterTypes.addVarchar(64, "part");
        parameterTypes.addInt("pass");
        parameterTypes.addInt("passes");
    }

    virtual TransformFunction *createTransformFunction(ServerInterface &srvInterface)
    { return vt_createFuncObject<VLoad>(srvInterface.allocator); }
};

RegisterFactory(VLoadFactory);
