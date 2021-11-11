#ifndef PG_MONTE_CARLO_CHECKERS_WINDOW_CUH
#define PG_MONTE_CARLO_CHECKERS_WINDOW_CUH

#include <SDL2/SDL.h>
#include <string>
#include <thread>
#include <condition_variable>
#include "Engine.cuh"

class Window {
private:
    int width, height;
    SDL_Window * window;
    SDL_Renderer * renderer;
    SDL_Texture * textures[5];

    int sleep_time = 0;
    bool click_to_continue = false;
    volatile bool waiting_for_click = true;

    bool running = false;

    bool moving = false;
    int chosen_piece{};
    int chosen_piece_position_x{};
    int chosen_piece_position_y{};

    std::thread * playing_thread{};
    std::condition_variable condition_variable;
    std::mutex mutex;
    Uint32 custom_event_type;

    Engine * engine_black = nullptr;
    Engine * engine_white = nullptr;
    GameState game_state;

    volatile bool awaiting_move = false;
public:
    Window(int width, int height);
    ~Window();
    void run();
    static void thread_run(Window * window);
    void push_custom_event() const;
    void paint();

    SDL_Texture * load_texture(const std::string& filename);
    static void print_error(const std::string& error_message);
    static void print_move(uint16_t move);

    bool finished() const;
    bool black_turn() const;
    uint8_t result() const;
};


#endif