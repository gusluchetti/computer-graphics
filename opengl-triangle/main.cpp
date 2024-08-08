#include <SDL2/SDL.h>
#include <SDL2/SDL_events.h>
#include <SDL2/SDL_video.h>
#include <cstdio>
#include <glad/glad.h>
#include <iostream>
#include <vector>

int gScreenHeight = 640;
int gScreenWidth = 640;
SDL_Window *glWindow = nullptr;
SDL_GLContext glContext = nullptr;
bool gQuit = false;

const std::string gVertexShaderSource =
    "#version 410 core\n"
    "in vec4 position;\n"
    "void main()\n"
    "{\n"
    "gl_Position = vec4(position.x, position.y, position.z, position.w);\n"
    "}\n";

const std::string gFragmentShaderSource =
    "#version 410 core\n"
    "out vec4 color;\n"
    "void main()\n"
    "{\n"
    "color = vec4(1.0f, 0.0f, 0.0f, 1.0f);\n"
    "}\n";

GLuint gVertexArrayObject = 0;
GLuint gVertexBufferObject = 0;
GLuint theProgram = 0;

GLuint CompileShader(GLuint type, const std::string &source) {
  std::string shaderSource;
  GLuint shaderObject = glCreateShader(type);

  const char *src = source.c_str();
  glShaderSource(shaderObject, 1, &src, nullptr);
  glCompileShader(shaderObject);

  return shaderObject;
}

void GetInfo() {
  std::cout << "Vendor: " << glGetString(GL_VENDOR) << "\n";
  std::cout << "Renderer: " << glGetString(GL_RENDERER) << "\n";
  std::cout << "Version: " << glGetString(GL_VERSION) << "\n";
  std::cout << "Shading Language: " << glGetString(GL_SHADING_LANGUAGE_VERSION)
            << "\n";
}

void MainLoop() {
  while (!gQuit) {
    SDL_Event e;

    while (SDL_PollEvent(&e) != 0) {
      if (e.type == SDL_QUIT) {
        std::cout << "bye!\n";
        gQuit = true;
        break;
      }
    }

    // pre draw
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    glViewport(0, 0, gScreenWidth, gScreenHeight);
    glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glUseProgram(theProgram);

    // draw
    glBindVertexArray(gVertexArrayObject);
    glBindBuffer(GL_ARRAY_BUFFER, gVertexBufferObject);

    glDrawArrays(GL_TRIANGLES, 0, 3);
    SDL_GL_SwapWindow(glWindow);
  }
}

int main(int argc, char *argv[]) {
  std::cout << "at main\n";
  std::cout << "init\n";

  const std::vector<GLfloat> vertexPosition{-0.8f, -0.8f, 0.0f, 0.8f, -0.8f,
                                            0.0f,  0.0f,  0.8f, 0.0f};

  // init program
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

  if (!gladLoadGLLoader(SDL_GL_GetProcAddress)) {
    std::cout << "glad could not be initialized\n";
    exit(1);
  }

  GetInfo();

  // vertex spec
  glGenVertexArrays(1, &gVertexArrayObject);
  glBindVertexArray(gVertexArrayObject);

  glGenBuffers(1, &gVertexBufferObject);
  glBindBuffer(GL_ARRAY_BUFFER, gVertexBufferObject);
  glBufferData(GL_ARRAY_BUFFER, vertexPosition.size() * sizeof(GLfloat),
               vertexPosition.data(), GL_STATIC_DRAW);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);

  glBindVertexArray(0);
  glDisableVertexAttribArray(0);

  theProgram = glCreateProgram();
  GLuint vertexShader = CompileShader(GL_VERTEX_SHADER, gVertexShaderSource);
  GLuint fragmentShader =
      CompileShader(GL_FRAGMENT_SHADER, gFragmentShaderSource);

  glAttachShader(theProgram, vertexShader);
  glAttachShader(theProgram, fragmentShader);
  glLinkProgram(theProgram);

  glValidateProgram(theProgram);

  MainLoop();

  // cleanup
  SDL_DestroyWindow(glWindow);
  SDL_Quit();

  std::cout << "quit\n";
  return 0;
}
