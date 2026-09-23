// vnode: tells which node holds which rows of the segmented table vvector.probe.
//   vvector_admin.vnode(k) OVER(PARTITION NODES) FROM vvector.probe
// Output (node_name, k): one row per node, with the smallest k stored on it.
// vload uses it to get exactly one probe row, and so one copy of the chunks, per node.
#include "Vertica.h"

using namespace Vertica;

static const char *const FN = "vnode";

class VNode : public TransformFunction
{
    virtual void processPartition(ServerInterface &srvInterface,
                                  PartitionReader &inputReader,
                                  PartitionWriter &outputWriter)
    {
        try {
            bool found = false;
            vint smallest = 0;
            do {
                if (inputReader.isNull(0)) continue;
                const vint k = inputReader.getIntRef(0);
                if (!found || k < smallest) { smallest = k; found = true; }
            } while (inputReader.next());
            if (!found) return;
            outputWriter.getStringRef(0).copy(srvInterface.getCurrentNodeName());
            outputWriter.setInt(1, smallest);
            outputWriter.next();
        } catch (std::exception &e) {
            vt_report_error(0, "%s: %s", FN, e.what());
        }
    }
};

class VNodeFactory : public TransformFunctionFactory
{
    virtual void getPrototype(ServerInterface &srvInterface, ColumnTypes &argTypes, ColumnTypes &returnType)
    {
        argTypes.addInt();
        returnType.addVarchar();
        returnType.addInt();
    }

    virtual void getReturnType(ServerInterface &srvInterface, const SizedColumnTypes &inputTypes,
                               SizedColumnTypes &outputTypes)
    {
        outputTypes.addVarchar(128, "node_name");
        outputTypes.addInt("k");
    }

    virtual TransformFunction *createTransformFunction(ServerInterface &srvInterface)
    { return vt_createFuncObject<VNode>(srvInterface.allocator); }
};

RegisterFactory(VNodeFactory);
