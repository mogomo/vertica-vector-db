// vvector.vversion() OVER(): library version, snapshot format version, build flags and the
// kernel code the CPU dispatcher picked on this node (x86_64: default, avx2 or avx512f).
// Thin adapter. The values come from src/engine/version.h.
#include "Vertica.h"
#include "../engine/kernels.h"
#include "../engine/version.h"

#include <cstring>

using namespace Vertica;

class VVersion : public TransformFunction
{
    virtual void processPartition(ServerInterface &srvInterface,
                                  PartitionReader &inputReader,
                                  PartitionWriter &outputWriter)
    {
        try {
            outputWriter.getStringRef(0).copy(vvector::LIBRARY_VERSION);
            outputWriter.setInt(1, vvector::FORMAT_VERSION);
            outputWriter.getStringRef(2).copy(std::string(vvector::BUILD_FLAGS) + " kernels=" + vvector::kernel_target());
            outputWriter.next();
        } catch (std::exception &e) {
            vt_report_error(0, "vversion: %s", e.what());
        }
    }
};

class VVersionFactory : public TransformFunctionFactory
{
    virtual void getPrototype(ServerInterface &srvInterface,
                              ColumnTypes &argTypes, ColumnTypes &returnType)
    {
        // No arguments.
        returnType.addVarchar();
        returnType.addInt();
        returnType.addVarchar();
    }

    virtual void getReturnType(ServerInterface &srvInterface,
                               const SizedColumnTypes &inputTypes,
                               SizedColumnTypes &outputTypes)
    {
        outputTypes.addVarchar(32, "library_version");
        outputTypes.addInt("format_version");
        outputTypes.addVarchar(256, "build_flags");
    }

    virtual TransformFunction *createTransformFunction(ServerInterface &srvInterface)
    { return vt_createFuncObject<VVersion>(srvInterface.allocator); }
};

RegisterFactory(VVersionFactory);
