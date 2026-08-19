#pragma once

#include "IActivityProfiler.h"

namespace libkineto {

class NVMLEnergyProfiler : public IActivityProfiler {
 public:
  explicit NVMLEnergyProfiler();
  ~NVMLEnergyProfiler() override = default;

  [[nodiscard]] auto name() const -> const std::string & override;
  [[nodiscard]] auto availableActivities() const -> const std::set<ActivityType> & override;
  auto configure(
      const std::set<libkineto::ActivityType> &activity_types,
      const Config &config) -> std::unique_ptr<libkineto::IActivityProfilerSession> override;
  auto configure(
      int64_t ts_ms, int64_t duration_ms,
      const std::set<libkineto::ActivityType> &activity_types,
      const Config &config) -> std::unique_ptr<libkineto::IActivityProfilerSession> override;
};

}  // namespace libkineto