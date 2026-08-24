// Rendering an op's metadata out of libtorch types.

#include "metadata.h"

#include <algorithm>
#include <ranges>
#include <sstream>

#include <c10/util/Logging.h>
#include <c10/util/ThreadLocalDebugInfo.h>
#ifdef USE_DISTRIBUTED
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#endif



namespace eprof {

namespace {

#ifdef USE_DISTRIBUTED
// The metadata keys a collective is described by. A chrome trace shows them
// verbatim, so the strings are the format.
constexpr auto kCommsName = "Collective name";
constexpr auto kDtype = "dtype";
constexpr auto kInMsgNelems = "In msg nelems";
constexpr auto kOutMsgNelems = "Out msg nelems";
constexpr auto kInSplit = "In split size";
constexpr auto kOutSplit = "Out split size";
constexpr auto kGlobalRankStart = "Global rank start";
constexpr auto kGlobalRankStride = "Global rank stride";
constexpr auto kGroupSize = "Group size";
constexpr auto kProcessGroupName = "Process Group Name";
constexpr auto kProcessGroupDesc = "Process Group Description";
constexpr auto kGroupRanks = "Process Group Ranks";
constexpr auto kRank = "Rank";
constexpr auto kP2pSrc = "Src Rank";
constexpr auto kP2pDst = "Dst Rank";
#endif  // USE_DISTRIBUTED
// Adopts a string from the Rust formatters and frees the original.

// Appends one shape to the i64 stream described in eprof-torch/src/inputs.rs.

}  // namespace

std::string ivalueToStr(const c10::IValue &val, bool isString) {
  std::stringstream ss;
  if (val.isNone()) {
    return "\"None\"";
  } else {
    ss.str("");
    if (isString) {
      ss << "\"";
    }
    ss << val;
    if (isString) {
      ss << "\"";
    }
    std::string mystr = ss.str();

    // For boolean the values that ivalue gives is "True" and "False" but
    // json only takes "true" and "false" so we convert the string to lower case
    if (val.isBool()) {
      for (char &c : mystr) {
        c = static_cast<char>(std::tolower(c));
      }
    }

    // A double quote can cause issues with the chrome tracing so force
    // all inputs to not contain more than the 2 we add in this function
    auto count = std::count(mystr.begin(), mystr.end(), '"');
    return count > 2 ? "\"None\"" : mystr;
  }
}

static constexpr int32_t kTruncatLength = 30;

template <typename ListLikeType>
requires requires(const ListLikeType::value_type& t) {
  { std::format("{}", t) } -> std::same_as<std::string>;
}
static inline std::string format_list(ListLikeType list, bool truncate,
                                      bool with_escaped_quotes = true) {
  auto join = [](auto begin, auto end) {
    std::string result;
    bool first = true;
    for (auto it : std::ranges::subrange(begin, end)) {
      if (!first) result += ", ";
      result += std::format("{}", it);
      first = false;
    }
    return result;
  };

  if (truncate && list.size() > kTruncatLength) {
    if (with_escaped_quotes == true) {
      return std::format("\"[{}, ...]\"", 
                        join(list.begin(), list.begin() + kTruncatLength));
    } else {
      return std::format("[{}, ...]", 
                        join(list.begin(), list.begin() + kTruncatLength));
    }
  }
  
  if (with_escaped_quotes == true) {
    return std::format("\"[{}]\"", 
                      join(list.begin(), list.end()));
  } else {
    return std::format("[{}]", 
                      join(list.begin(), list.end()));
  }
}

std::unordered_map<std::string, std::string> saveNcclMeta(
    // @lint-ignore CLANGTIDY
    const at::RecordFunction &fn) {
  std::unordered_map<std::string, std::string> map;
#ifdef USE_DISTRIBUTED
  auto debugInfo = dynamic_cast<::torch::ParamCommsDebugInfo *>(
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::PARAM_COMMS_INFO));

  {
    if (debugInfo == nullptr) {
      LOG(WARNING) << "ParamCommsDebugInfo not available for function: "
                   << fn.name();
      return map;
    }
    auto &collective_name = debugInfo->getCollectiveName();
    map.emplace(kCommsName, std::format("\"{}\"", collective_name));
    map.emplace(kDtype,
                std::format("\"{}\"", c10::toString(debugInfo->getDType())));
    map.emplace(kInMsgNelems, std::to_string(debugInfo->getInMessageNelems()));
    map.emplace(kOutMsgNelems,
                std::to_string(debugInfo->getOutMessageNelems()));

    auto &inSplitSizes = debugInfo->getInputSplitSizes();
    map.emplace(kInSplit, format_list(inSplitSizes, /*truncate=*/true));

    auto &outSplitSizes = debugInfo->getOutputSplitSizes();
    map.emplace(kOutSplit, format_list(outSplitSizes, /*truncate=*/true));

    auto globalRankStart = debugInfo->getGlobalRankStart();
    if (globalRankStart >= 0) {
      map.emplace(kGlobalRankStart, std::to_string(globalRankStart));
    }
    auto globalRankStride = debugInfo->getGlobalRankStride();
    if (globalRankStride > 0) {
      map.emplace(kGlobalRankStride, std::to_string(globalRankStride));
    }
    map.emplace(kGroupSize, std::to_string(debugInfo->getWorldSize()));
    auto &group_name = debugInfo->getProcessGroupName();
    if (!group_name.empty()) {
      map.emplace(kProcessGroupName, std::format("\"{}\"", group_name));
    }
    auto &group_desc = debugInfo->getProcessGroupDesc();
    if (!group_desc.empty()) {
      map.emplace(kProcessGroupDesc, std::format("\"{}\"", group_desc));
    }
    auto &groupRanks = debugInfo->getGroupRanks();
    map.emplace(kGroupRanks, format_list(groupRanks, /*truncate=*/true));

    auto rank = debugInfo->getRank();
    map.emplace(kRank, std::to_string(rank));
    int nRanks = static_cast<int>(groupRanks.size());
    if (collective_name == "send") {
      if (rank >= 0 && rank < nRanks) {
        map.emplace(kP2pDst, std::to_string(groupRanks[rank]));
      }
    } else if (collective_name == "recv") {
      if (rank >= 0 && rank < nRanks) {
        map.emplace(kP2pSrc, std::to_string(groupRanks[rank]));
      }
    }
  }

#endif  // USE_DISTRIBUTED
  return map;
}

}  // namespace eprof
