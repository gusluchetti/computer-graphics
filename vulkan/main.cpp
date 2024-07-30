#include <vulkan/vulkan.h>

#include <cstdlib>
#include <iostream>

class HelloTriangleApplication {
public:
  void run() {
    initVulkan();
    mainLoop();
    cleanup();
  }

private:
  void initVulkan() {}

  void mainLoop() {}

  void cleanup() {}
};

int main() {
  std::cout << "init main\n";
  HelloTriangleApplication app;

  try {
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "quit main\n";
  return EXIT_SUCCESS;
}
