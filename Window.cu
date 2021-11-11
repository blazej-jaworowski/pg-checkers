#include "Window.cuh"

#include <iostream>

Window::Window(int width, int height) : width(width), height(height) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL initialisation failed." << std::endl;
    }

    custom_event_type = SDL_RegisterEvents(1);
    if (custom_event_type == (Uint32) -1) print_error("Error registering custom event");

    window = SDL_CreateWindow("Checkers", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, 0);
    if (window == nullptr) print_error("Error creating window");
    renderer = SDL_CreateRenderer(window, -1, 0);
    if (renderer == nullptr) print_error("Error creating renderer");

    textures[0] = load_texture("chessboard.bmp");
    textures[1] = load_texture("white_man.bmp");
    textures[2] = load_texture("black_man.bmp");
    textures[3] = load_texture("white_king.bmp");
    textures[4] = load_texture("black_king.bmp");

    engine_black = new Engine();
//    engine_black->time_per_move = 10;
    engine_white = new Engine();
//    engine_white->time_per_move = 100;
}

void Window::thread_run(Window *window) {
    while (window->running && !window->finished()) {
        if (window->black_turn()) {
            if (window->engine_black != nullptr) {
                while (window->black_turn() && window->running) {
                    uint16_t move = window->engine_black->get_move();
                    if (move == 0xFFFF) break;
                    if (window->click_to_continue) {
                        while (window->waiting_for_click) {
                            SDL_Delay(10);
                        }
                        window->waiting_for_click = true;
                    } else {

                        SDL_Delay(window->sleep_time);
                    }
                    window->engine_black->play_move(move);
                    if(window->engine_white != nullptr) window->engine_white->play_move(move);
                    window->game_state.play_move(move);
                    window->game_state.calculate_game_state();
                    window->push_custom_event();
                    print_move(move);
                }
                std::cout << std::endl;
                continue;
            }

            window->awaiting_move = true;
            std::unique_lock<std::mutex> lock(window->mutex);
            window->condition_variable.wait(lock);
            continue;
        }

        if (window->engine_white != nullptr) {
            while (!window->black_turn() && window->running) {
                uint16_t move = window->engine_white->get_move();
                if (move == 0xFFFF) break;
                if (window->click_to_continue) {
                    while (window->waiting_for_click) {
                        SDL_Delay(10);
                    }
                    window->waiting_for_click = true;
                } else {

                    SDL_Delay(window->sleep_time);
                }
                window->engine_white->play_move(move);
                if(window->engine_black != nullptr) window->engine_black->play_move(move);
                window->game_state.play_move(move);
                window->game_state.calculate_game_state();
                window->push_custom_event();
                print_move(move);
            }
            std::cout << std::endl;
            continue;
        }

        window->awaiting_move = true;
        std::unique_lock<std::mutex> lock(window->mutex);
        window->condition_variable.wait(lock);
    }
    std::cout << (window->result() == 1 ? "Draw!" : (window->result() == 0 ?
                  "Black won!" :
                  "White won!")) << std::endl;
}

void Window::run() {
    running = true;
    playing_thread = new std::thread(thread_run, this);
    SDL_Event event;
    while (running) {
        if (!SDL_PollEvent(&event)) continue;
        switch (event.type) {
            case SDL_QUIT:
                running = false;
                break;
            case SDL_MOUSEBUTTONDOWN:
                if (waiting_for_click) waiting_for_click = false;
                if (moving || !awaiting_move) break;
                if (event.button.x < 0 || event.button.x >= 512 ||
                    event.button.y < 0 || event.button.y >= 512 ||
                    (event.button.x / 64 + event.button.y / 64) % 2 == 0)
                    break;
                chosen_piece_position_x = event.button.x;
                chosen_piece_position_y = event.button.y;
                chosen_piece = GameState::xy_to_index(event.button.x / 64, event.button.y / 64);
                if (game_state.board_state[chosen_piece] != 0) moving = true;
                break;
            case SDL_MOUSEMOTION:
                if (!moving) break;
                chosen_piece_position_x = event.motion.x;
                chosen_piece_position_y = event.motion.y;
                break;
            case SDL_MOUSEBUTTONUP:
                if (!moving) break;
                moving = false;
                if ((event.button.x / 64 + event.button.y / 64) % 2 == 0) break;
                uint16_t move = GameState::create_move(chosen_piece,
                                                       GameState::xy_to_index(chosen_piece_position_x / 64,
                                                                              chosen_piece_position_y / 64));
                if (!game_state.move_valid(move)) break;
                if(engine_black != nullptr) engine_black->play_move(move);
                if(engine_white != nullptr) engine_white->play_move(move);
                game_state.play_move(move);
                game_state.calculate_game_state();
                print_move(move);
                std::cout << std::endl;

                condition_variable.notify_all();
                break;
        }

        paint();
    }
    playing_thread->join();
}

void Window::paint() {
    SDL_RenderClear(renderer);
    SDL_Rect rect = {0, 0, width, height};
    SDL_RenderCopy(renderer, textures[0], &rect, &rect);

    SDL_Rect piece_rect = {0, 0, 64, 64};
    rect = {0, 0, 64, 64};

    const uint8_t *board_state = game_state.board_state;
    for (int i = 0; i < 32; i++) {
        if (board_state[i] == 0) continue;
        if (moving && chosen_piece == i) continue;

        rect.x = GameState::index_to_x(i) * 64;
        rect.y = GameState::index_to_y(i) * 64;

        SDL_RenderCopy(renderer, textures[board_state[i]], &piece_rect, &rect);
    }

    if (moving) {
        rect.x = chosen_piece_position_x - 32;
        rect.y = chosen_piece_position_y - 32;
        SDL_RenderCopy(renderer, textures[board_state[chosen_piece]], &piece_rect, &rect);
    }

    SDL_RenderPresent(renderer);
}

Window::~Window() {
    for (int i = 0; i < 5; i++) {
        SDL_DestroyTexture(textures[i]);
    }
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

SDL_Texture *Window::load_texture(const std::string &filename) {
    SDL_Surface *surface = SDL_LoadBMP(filename.c_str());
    if (surface == nullptr) print_error("Error creating surface from BMP");
    SDL_Texture *r = SDL_CreateTextureFromSurface(renderer, surface);
    if (r == nullptr) print_error("Error creating texture");
    SDL_FreeSurface(surface);
    return r;
}

void Window::print_error(const std::string &error_message) {
    std::cerr << error_message << ": " << SDL_GetError() << std::endl;
}


void Window::print_move(uint16_t move) {
    uint8_t from = GameState::get_from(move);
    uint8_t to = GameState::get_to(move);
    std::cout << GameState::index_to_x(from) << ", " << GameState::index_to_y(from) <<
              " -> " << GameState::index_to_x(to) << ", " << GameState::index_to_y(to) << std::endl;
}

void Window::push_custom_event() const {
    SDL_Event event;
    SDL_zero(event);
    event.type = custom_event_type;
    SDL_PushEvent(&event);
}

bool Window::black_turn() const {
    return game_state.black_turn;
}

bool Window::finished() const {
    return game_state.finished;
}

uint8_t Window::result() const {
    return game_state.result;
}
