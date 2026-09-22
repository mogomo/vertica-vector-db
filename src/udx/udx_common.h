// Helpers shared by the vvector UDx adapters.
#ifndef VVECTOR_UDX_COMMON_H
#define VVECTOR_UDX_COMMON_H

#include "Vertica.h"
#include "Arrays/Accessors.h"
#include "../engine/cache.h"
#include "../engine/snapshot.h"

#include <string>
#include <vector>

namespace vvector_udx {

using Vertica::BaseDataOID;     // Float8OID is a macro that names this type without its namespace

// Input columns of the query functions (draft: the interface may still change):
//   qid INT, qvec ARRAY[FLOAT]                          a request row: one query vector
//   id INT, vec ARRAY[FLOAT], del BOOLEAN, ver INT      a journal (delta) row
//   snapshot_id INT                                     on every row of a delta view: the snapshot the view belongs to
enum InputColumn { COL_QID = 0, COL_QVEC, COL_ID, COL_VEC, COL_DEL, COL_VER, COL_SNAPSHOT_ID, COL_COUNT };

inline void add_query_input(Vertica::ColumnTypes &argTypes)
{
    argTypes.addInt();                          // qid
    argTypes.addArrayType(Float8OID);           // qvec
    argTypes.addInt();                          // id
    argTypes.addArrayType(Float8OID);           // vec
    argTypes.addBool();                         // del
    argTypes.addInt();                          // ver
    argTypes.addInt();                          // snapshot_id
}

// Reads an ARRAY[FLOAT] cell into out as float32. The caller checks the cell for NULL first.
// Throws std::runtime_error on a NULL element.
inline void read_vector(Vertica::PartitionReader &in, size_t col, std::vector<float> &out)
{
    out.clear();
    Vertica::Array::ArrayReader a = in.getArrayRef(col);
    for (; a->hasData(); a->next()) {
        if (a->isNull(0)) throw std::runtime_error("a vector has a NULL element");
        out.push_back(static_cast<float>(a->getFloatRef(0)));
    }
}

// cache_dir precedence: function parameter, then session parameter
// (ALTER SESSION SET UDPARAMETER FOR vvector cache_dir = '...'), then the default.
inline std::string resolve_cache_dir(Vertica::ServerInterface &srvInterface)
{
    Vertica::ParamReader params = srvInterface.getParamReader();
    if (params.containsParameter("cache_dir")) return params.getStringRef("cache_dir").str();
    Vertica::ParamReader session = srvInterface.getUDSessionParamReader("library");
    if (session.containsParameter("cache_dir")) return session.getStringRef("cache_dir").str();
    return vvector::DEFAULT_CACHE_DIR;
}

// The index_name parameter, checked. Reports an error naming fn when it is missing or not valid.
inline std::string read_index_name(const char *fn, Vertica::ServerInterface &srvInterface)
{
    Vertica::ParamReader params = srvInterface.getParamReader();
    if (!params.containsParameter("index_name"))
        vt_report_error(0, "%s: parameter index_name is required", fn);
    const std::string name = params.getStringRef("index_name").str();
    if (!vvector::valid_index_name(name))
        vt_report_error(0, "%s: index name '%s' is not valid: use letters, digits and underscore", fn, name.c_str());
    return name;
}

inline void add_common_parameters(Vertica::SizedColumnTypes &parameterTypes)
{
    parameterTypes.addVarchar(128, "index_name");
    parameterTypes.addVarchar(1024, "cache_dir");
}

} // namespace vvector_udx

#endif
