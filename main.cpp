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
    "#version 410 core"
    "in vec4 position;"
    "void main()"
    "{"
    "gl_Position = vec4(position.x, position.y, position.z, position.w);"
    "}";

const std::string gFragmentShaderSource =
    "#version 410 core"
    "out vec4 color;"
    "void main()"
    "{"
    "color = vec4(1.0f, 0.0f, 0.0f, 1.0f);"
    "}";

GLuint gVertexArrayObject = 0;
GLuint gVertexBufferObject = 0;

GLuint gGraphicsPipelineShaderProgram = 0;

void debugMessage(GLenum source, GLenum type, GLuint id, GLenum severity,
                  GLsizei length, const GLchar *message,
                  const void *userParam) {
  // Print, log, whatever based on the enums and message
  std::string message_str(message, length);
  std::cout << message_str << '\n';
}

void VertexSpec() {
  const std::vector<GLfloat> vertexPosition{-0.8f, -0.8f, 0.0f, 0.8f, -0.8f,
                                            0.0f,  0.0f,  0.8f, 0.0f};

  glGenVertexArrays(1, &gVertexArrayObject);
  glBindVertexArray(gVertexArrayObject);

  glGenBuffers(1, &gVertexBufferObject);
  glBindBuffer(GL_ARRAY_BUFFER, gVertexBufferObject);
  glBufferData(GL_ARRAY_BUFFER, vertexPosition.size() * sizeof(GLfloat),
               vertexPosition.data(), GL_STATIC_DRAW);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);

  // glBindVertexArray(0);
  // glDisableVertexAttribArray(0);
}

GLuint CompileShader(GLuint type, const std::string &source) {
  std::string shaderSource;
  GLuint shaderObject = glCreateShader(type);

  const char *src = source.c_str();
  glShaderSource(shaderObject, 1, &src, nullptr);
  glCompileShader(shaderObject);

  return shaderObject;
}

void CreateGraphicsPipeline() {
  GLuint programObject = glCreateProgram();
  GLuint vertexShader = CompileShader(GL_VERTEX_SHADER, gVertexShaderSource);
  GLuint fragmentShader =
      CompileShader(GL_FRAGMENT_SHADER, gFragmentShaderSource);

  glAttachShader(programObject, vertexShader);
  glAttachShader(programObject, fragmentShader);
  glLinkProgram(programObject);
  glValidateProgram(programObject);
  // glDetachShader, glDeleteShader;
}

void GetInfo() {
  std::cout << "Vendor: " << glGetString(GL_VENDOR) << "\n";
  std::cout << "Renderer: " << glGetString(GL_RENDERER) << "\n";
  std::cout << "Version: " << glGetString(GL_VERSION) << "\n";
  std::cout << "Shading Language: " << glGetString(GL_SHADING_LANGUAGE_VERSION)
            << "\n";
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

  if (!gladLoadGLLoader(SDL_GL_GetProcAddress)) {
    std::cout << "glad could not be initialized\n";
    exit(1);
  }

  GetInfo();
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

void PreDraw() {
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);

  glViewport(0, 0, gScreenWidth, gScreenHeight);
  glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
  glUseProgram(gGraphicsPipelineShaderProgram);
}

void Draw() {
  glBindVertexArray(gVertexArrayObject);
  glBindBuffer(GL_ARRAY_BUFFER, gVertexBufferObject);

  glDrawArrays(GL_TRIANGLES, 0, 3);
}

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

  glEnable(GL_DEBUG_OUTPUT);
  glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
  glDebugMessageCallback(debugMessage, NULL);
  glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL,
                        GL_TRUE);

  VertexSpec();
  CreateGraphicsPipeline();
  MainLoop();
  Cleanup();
  std::cout << "quit\n";

  return 0;
}
