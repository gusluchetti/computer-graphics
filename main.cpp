#include <SDL2/SDL.h>
#include <SDL2/SDL_events.h>
#include <SDL2/SDL_video.h>
#include <cstdio>
#include <iostream>
#include <glad/glad.h>

int gScreenHeight = 640;
int gScreenWidth = 480;
SDL_Window *glWindow = nullptr;
SDL_GLContext glContext = nullptr;
bool gQuit = false;

void GetInfo() {
    std::cout << "Vendor: " << glGetString(GL_VENDOR) << "\n";
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << "\n";
    std::cout << "Version: " << glGetString(GL_VERSION) << "\n";
    std::cout << "Shading Language: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << "\n";
}

void InitProgram() {
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    std::cout << "SDL2 could not initialize video subsys\n";
    exit(1);
  }

  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);

  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

  glWindow = SDL_CreateWindow("OpenGL Window", 0, 0, gScreenWidth,
                              gScreenHeight, SDL_WINDOW_OPENGL);
  if (glWindow == nullptr) {
    std::cout << "Window could not be initialized\n";
    exit(1);
  }

  glContext = SDL_GL_CreateContext(glWindow);
  if (glContext == nullptr) {
    std::cout << "OpenGL not available\n";
    exit(1);
  }
}

void Input() {
  SDL_Event e;

  while (SDL_PollEvent(&e) != 0) {
    if (e.type == SDL_QUIT) {
      std::cout << "bye!\n";
      gQuit = true;
      break;
    }
  }
}

void PreDraw() {}

void Draw() {}

void MainLoop() {
  while (!gQuit) {
    Input();
    PreDraw();
    Draw();
    SDL_GL_SwapWindow(glWindow);
  }
}

void Cleanup() {
  SDL_DestroyWindow(glWindow);
  SDL_Quit();
}

int main(int argc, char *argv[]) {
  std::cout << "at main\n";
  std::cout << "init\n";
  InitProgram();
  MainLoop();
  Cleanup();
  std::cout << "quit\n";

  return 0;
}
