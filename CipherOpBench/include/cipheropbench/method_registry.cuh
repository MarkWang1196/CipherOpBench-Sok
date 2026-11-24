#pragma once
#include "cipheropbench/operators.cuh"
#include <map>
#include <memory>

namespace cipheropbench {
// Methods own operator dispatch, supported shapes and numerical contracts.
// The benchmark runner owns timing and output records. A new implementation
// can subclass TransformerOperators and replace individual virtual operators.
struct MethodDescriptor {
  std::string name, backend, packing_contract, source_revision;
  Shape shape;
};
class MethodRegistry {
public:
  using Factory=std::function<std::unique_ptr<TransformerOperators>(Runtime&)>;
  static MethodRegistry& instance() { static MethodRegistry registry; return registry; }
  void add(MethodDescriptor descriptor, Factory factory) {
    auto name=descriptor.name;
    TORCH_CHECK(!factories_.count(name),"duplicate method: ",name);
    descriptors_.emplace(name,std::move(descriptor));factories_.emplace(name,std::move(factory));
  }
  std::unique_ptr<TransformerOperators> create(const std::string& name,Runtime& runtime) const {
    TORCH_CHECK(factories_.count(name),"unknown method: ",name);
    descriptors_.at(name).shape.validate_nexus();
    return factories_.at(name)(runtime);
  }
  const MethodDescriptor& descriptor(const std::string& name) const { return descriptors_.at(name); }
private:
  std::map<std::string,MethodDescriptor> descriptors_;
  std::map<std::string,Factory> factories_;
};
inline void register_nexus_method() {
  MethodRegistry::instance().add({"nexus","PhantomBootstrapping","nexus_128x768_row_headpair_diagonal","23288730907fdb7a8ff7e4683efca35164f3cb8d",{}},
    [](Runtime& runtime){return std::make_unique<TransformerOperators>(runtime);});
}
}
