#include "glomap/exe/multi_global_mapper.h"

#include <colmap/util/logging.h>

#include <iostream>

namespace {

typedef std::function<int(int, char**)> command_func_t;
int ShowHelp(
    const std::vector<std::pair<std::string, command_func_t>>& commands) {
  std::cout << "MGSfM -- Multi-Camera Global Structure-from-Motion" << std::endl
            << std::endl;

#ifdef GLOMAP_CUDA_ENABLED
  std::cout << "This version was compiled with CUDA!" << std::endl << std::endl;
#else
  std::cout << "This version was NOT compiled CUDA!" << std::endl << std::endl;
#endif

  std::cout << "Usage:" << std::endl;
  std::cout
      << "  mgsfm multi_mapper --database_path DATABASE --output_path MODEL"
      << std::endl;

  std::cout << "Available commands:" << std::endl;
  std::cout << "  help" << std::endl;
  for (const auto& command : commands) {
    std::cout << "  " << command.first << std::endl;
  }
  std::cout << std::endl;

  return EXIT_SUCCESS;
}
}  // namespace

int main(int argc, char** argv) {
  colmap::InitializeGlog(argv);
  FLAGS_alsologtostderr = true;

  std::vector<std::pair<std::string, command_func_t>> commands;
  commands.emplace_back("multi_mapper", &MGSfM::RunMGSfM);

  if (argc == 1) {
    return ShowHelp(commands);
  }

  const std::string command = argv[1];
  if (command == "help" || command == "-h" || command == "--help") {
    return ShowHelp(commands);
  } else {
    command_func_t matched_command_func = nullptr;
    for (const auto& command_func : commands) {
      if (command == command_func.first) {
        matched_command_func = command_func.second;
        break;
      }
    }
    if (matched_command_func == nullptr) {
      std::cout << "Command " << command << " not recognized. "
                << "To list the available commands, run `glomap help`."
                << std::endl;
      return EXIT_FAILURE;
    } else {
      int command_argc = argc - 1;
      char** command_argv = &argv[1];
      command_argv[0] = argv[0];
      return matched_command_func(command_argc, command_argv);
    }
  }

  return ShowHelp(commands);
}
