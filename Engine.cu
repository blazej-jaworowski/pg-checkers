#include "Engine.cuh"

#include <cstring>

Engine::Engine(const uint8_t * board_state) {
    memcpy(this->board_state, board_state, 32);
}

uint16_t Engine::get_move() {
    return create_move(13, 17);
}

uint16_t Engine::create_move(uint8_t from, uint8_t to) {
    return (from << 8) + to;
}

bool Engine::play_move(uint16_t move) {
    uint8_t from = get_from(move);
    uint8_t to = get_to(move);
    if(board_state[to] == board_state[from]) return false;
    board_state[to] = board_state[from];
    board_state[from] = 0;
    return true;
}

uint8_t Engine::get_from(uint16_t move) {
    return move >> 8;
}

uint8_t Engine::get_to(uint16_t move) {
    return move & 0xff;
}

const uint8_t *Engine::get_board_state() {
    return board_state;
}

Engine::Engine() {
    memset(board_state, 2, 12);
    memset(board_state + 12, 0, 8);
    memset(board_state + 20, 1, 12);
}

