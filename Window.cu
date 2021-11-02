#include "Window.cuh"

#include <iostream>

Window::Window(int width, int height) : width(width), height(height) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL initialisation failed." << std::endl;
    }

    window = SDL_CreateWindow("Checkers", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, 0);
    if (window == NULL) print_error("Error creating window");
    renderer = SDL_CreateRenderer(window, -1, 0);
    if (renderer == NULL) print_error("Error creating renderer");

    board_texture = load_texture("chessboard.bmp");
    white_texture = load_texture("white.bmp");
    black_texture = load_texture("black.bmp");
}

void Window::run() {
    SDL_Event event;
    running = true;
    while (running) {
        if (!SDL_PollEvent(&event)) continue;
        switch (event.type) {
            case SDL_QUIT:
                running = false;
                break;
            case SDL_MOUSEBUTTONDOWN:
                if (moving) break;
                if (event.button.x < 0 || event.button.x >= 512 ||
                    event.button.y < 0 || event.button.y >= 512 ||
                    (event.button.x / 64 + event.button.y / 64) % 2 == 0)
                    break;
                chosen_piece_position_x = event.button.x;
                chosen_piece_position_y = event.button.y;
                chosen_piece = xy_to_index(event.button.x / 64, event.button.y / 64);
                if (engine.get_board_state()[chosen_piece] != 0) moving = true;
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
                engine.play_move(Engine::create_move(chosen_piece, xy_to_index(chosen_piece_position_x / 64,
                                                                               chosen_piece_position_y / 64)));
                break;
        }

        paint();
    }
}

void Window::paint() {
    SDL_RenderClear(renderer);
    SDL_Rect rect = {0, 0, width, height};
    SDL_RenderCopy(renderer, board_texture, &rect, &rect);

    SDL_Rect piece_rect = {0, 0, 64, 64};
    rect = {0, 0, 64, 64};

    const uint8_t *board_state = engine.get_board_state();
    for (int i = 0; i < 32; i++) {
        if (board_state[i] == 0) continue;
        if (moving && chosen_piece == i) continue;

        rect.x = index_to_x(i) * 64;
        rect.y = index_to_y(i) * 64;

        SDL_RenderCopy(renderer, board_state[i] == 1 ? white_texture : black_texture, &piece_rect, &rect);
    }

    if (moving) {
        rect.x = chosen_piece_position_x - 32;
        rect.y = chosen_piece_position_y - 32;
        SDL_RenderCopy(renderer, board_state[chosen_piece] == 1 ? white_texture : black_texture, &piece_rect, &rect);
    }

    SDL_RenderPresent(renderer);
}

Window::~Window() {
    SDL_DestroyTexture(board_texture);
    SDL_DestroyTexture(white_texture);
    SDL_DestroyTexture(black_texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

SDL_Texture *Window::load_texture(const std::string &filename) {
    SDL_Surface *surface = SDL_LoadBMP(filename.c_str());
    if (surface == NULL) print_error("Error creating surface from BMP");
    SDL_Texture *r = SDL_CreateTextureFromSurface(renderer, surface);
    if (r == NULL) print_error("Error creating texture");
    SDL_FreeSurface(surface);
    return r;
}

void Window::print_error(const std::string &error_message) {
    std::cerr << error_message << ": " << SDL_GetError() << std::endl;
}

int Window::index_to_x(int index) {
    return (index % 4) * 2 + (index % 8 < 4);
}

int Window::index_to_y(int index) {
    return index / 4;
}

int Window::xy_to_index(int x, int y) {
    return x / 2 + y * 4;
}
