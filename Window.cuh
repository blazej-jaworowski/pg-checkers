#ifndef PG_MONTE_CARLO_CHECKERS_WINDOW_CUH
#define PG_MONTE_CARLO_CHECKERS_WINDOW_CUH

#include <SDL2/SDL.h>
#include <string>
#include "Engine.cuh"

class Window {
private:
    int width, height;
    SDL_Window * window;
    SDL_Renderer * renderer;
    SDL_Texture * board_texture;
    SDL_Texture * white_texture;
    SDL_Texture * black_texture;

    Engine engine;
    bool running = false;

    bool moving = false;
    int chosen_piece;
    int chosen_piece_position_x;
    int chosen_piece_position_y;
public:
    Window(int width, int height);
    ~Window();
    void run();
    void paint();

    SDL_Texture * load_texture(const std::string& filename);
    static void print_error(const std::string& error_message);
    static inline int index_to_x(int index);
    static inline int index_to_y(int index);
    static inline int xy_to_index(int x, int y);
};


#endif