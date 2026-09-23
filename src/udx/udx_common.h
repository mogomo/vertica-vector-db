// Helpers shared by the vvector UDx adapters.
// Errors: the adapters throw std::runtime_error and report it once, in their catch block, as
// "<function>: <cause>".
#ifndef VVECTOR_UDX_COMMON_H
#define VVECTOR_UDX_COMMON_H

#include "Vertica.h"
#include "Arrays/Accessors.h"
#include "../engine/cache.h"
#include "../engine/snapshot.h"

#include <cerrno>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace vvector_udx {

using Vertica::BaseDataOID;     // Float8OID is a macro that names this type without its namespace

[[noreturn]] inline void fail(const std::string &why) { throw std::runtime_error(why); }

// Input columns of vsearch. The role of a row is given by its NULLs:
//   qid, qvec set                       a query row
//   id, vec set, del false              journal add or replace (delta view)
//   id set, del true                    journal delete (delta view)
//   id set, vec and del NULL            allow-list member (filtered search, milestone M5)
//   only snapshot_id set                sentinel: the snapshot the view belongs to
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

// Reads the ARRAY[FLOAT] cell of column col (the caller checks the cell for NULL first). Calls
// room(n) for a place of n floats and converts the elements to float32 into it. A 1-D FLOAT
// array is one contiguous run of float8 values in the input block: it is read through a pointer,
// not with a call per element. Throws on a NULL element and on a value that is not a finite
// float32 (NaN, Infinity, or beyond +-3.4e38). `what` names the row in the message.
template <class Room>
std::uint32_t read_floats(Vertica::PartitionReader &in, std::size_t col, Room room, const std::string &what)
{
    Vertica::Array::ArrayReader a = in.getArrayRef(col);
    const int n = a->getNumRows();
    float *out = room(static_cast<std::uint32_t>(n));
    if (n <= 0) return 0;
    bool bad = false;
    const bool contiguous = a->getColStride(0) == static_cast<int>(sizeof(Vertica::vfloat));
    if (contiguous) {
        const Vertica::vfloat *p = a->getFloatPtr(0);
        for (int i = 0; i < n; ++i) {
            const float f = static_cast<float>(p[i]);
            out[i] = f;
            bad |= !(std::fabs(f) <= FLT_MAX);
        }
    } else {
        for (int i = 0; a->hasData(); a->next(), ++i) {
            const float f = static_cast<float>(a->getFloatRef(0));
            out[i] = f;
            bad |= !(std::fabs(f) <= FLT_MAX);
        }
    }
    if (bad) {
        for (int i = 0; i < n; ++i)
            if (!(std::fabs(out[i]) <= FLT_MAX)) {
                if (!contiguous) fail(what + ": element " + std::to_string(i + 1) + " is NULL or not a finite float32 value");
                const Vertica::vfloat v = a->getFloatPtr(0)[i];
                if (Vertica::vfloatIsNull(v)) fail(what + " has a NULL element");
                fail(what + ": element " + std::to_string(i + 1) + " is not a finite float32 value (" + std::to_string(v) + ")");
            }
    }
    return static_cast<std::uint32_t>(n);
}

// ---- parameters

// cache_dir: function parameter, then session parameter
// (ALTER SESSION SET UDPARAMETER FOR vvector cache_dir = '...'), then the default.
inline std::string resolve_cache_dir(Vertica::ServerInterface &srvInterface)
{
    Vertica::ParamReader params = srvInterface.getParamReader();
    if (params.containsParameter("cache_dir")) return params.getStringRef("cache_dir").str();
    Vertica::ParamReader session = srvInterface.getUDSessionParamReader("library");
    if (session.containsParameter("cache_dir")) return session.getStringRef("cache_dir").str();
    return vvector::DEFAULT_CACHE_DIR;
}

// The index_name parameter, checked.
inline std::string read_index_name(Vertica::ServerInterface &srvInterface)
{
    Vertica::ParamReader params = srvInterface.getParamReader();
    if (!params.containsParameter("index_name")) fail("parameter index_name is required");
    const std::string name = params.getStringRef("index_name").str();
    if (!vvector::valid_index_name(name)) fail("index name '" + name + "' is not valid: use letters, digits and underscore");
    return name;
}

inline void add_common_parameters(Vertica::SizedColumnTypes &parameterTypes)
{
    parameterTypes.addVarchar(128, "index_name");
    parameterTypes.addVarchar(1024, "cache_dir");
}

// Tuning values with the precedence: function parameter, then session parameter
// (ALTER SESSION SET UDPARAMETER FOR vvector <name> = '...'), then the index default
// (set_index_options, from the OPTIONS file of the node cache), then the built-in default.
class Settings {
public:
    Settings(Vertica::ServerInterface &srv, const vvector::IndexOptions &index)
        : params_(srv.getParamReader()), session_(srv.getUDSessionParamReader("library")), index_(index) {}

    std::string text(const char *name, const std::string &builtin)
    {
        if (params_.containsParameter(name)) return params_.getStringRef(name).str();
        std::string v, from;
        return lookup(name, v, from) ? v : builtin;
    }

    std::int64_t integer(const char *name, std::int64_t builtin)
    {
        if (params_.containsParameter(name)) return params_.getIntRef(name);
        std::string v, from;
        if (!lookup(name, v, from)) return builtin;
        char *end = nullptr;
        errno = 0;
        const long long x = std::strtoll(v.c_str(), &end, 10);
        if (v.empty() || *end || errno) fail(from + " " + name + " = '" + v + "' is not an integer");
        return x;
    }

    double real(const char *name, double builtin)
    {
        if (params_.containsParameter(name)) return params_.getFloatRef(name);
        std::string v, from;
        if (!lookup(name, v, from)) return builtin;
        char *end = nullptr;
        const double x = std::strtod(v.c_str(), &end);
        if (v.empty() || *end || !std::isfinite(x)) fail(from + " " + name + " = '" + v + "' is not a number");
        return x;
    }

    bool boolean(const char *name, bool builtin)
    {
        if (params_.containsParameter(name)) return params_.getBoolRef(name) == Vertica::vbool_true;
        std::string v, from;
        if (!lookup(name, v, from)) return builtin;
        if (v == "true" || v == "t" || v == "1" || v == "yes") return true;
        if (v == "false" || v == "f" || v == "0" || v == "no") return false;
        fail(from + " " + name + " = '" + v + "' is not true or false");
    }

private:
    bool lookup(const char *name, std::string &v, std::string &from)
    {
        if (session_.containsParameter(name)) { v = session_.getStringRef(name).str(); from = "session parameter"; return true; }
        auto it = index_.find(name);
        if (it != index_.end()) { v = it->second; from = "index default"; return true; }
        return false;
    }
    Vertica::ParamReader params_, session_;
    const vvector::IndexOptions &index_;
};

} // namespace vvector_udx

#endif
