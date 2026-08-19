#pragma once

#include "IActivityProfiler.h"

namespace libkineto {

class NVMLEnergyProfiler : public IActivityProfiler {
 public:
  explicit NVMLEnergyProfiler();
  ~NVMLEnergyProfiler() override = default;

  const std::string &name() const override;
  const std::set<ActivityType> &availableActivities() const override;
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      const std::set<libkineto::ActivityType> &activity_types,
      const Config &config) override;
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      int64_t ts_ms,
      int64_t duration_ms,
      const std::set<libkineto::ActivityType>& activity_types,
      const Config& config) override;
};

}