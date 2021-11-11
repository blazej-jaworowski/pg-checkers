#include <iostream>
#include "GameState.cuh"
#include "Window.cuh"

GameState::GameState() {
    memset(board_state, 2, 12);
    memset(board_state + 12, 0, 8);
    memset(board_state + 20, 1, 12);
    black_piece_count = 12;
    white_piece_count = 12;
}

bool GameState::take_piece(uint8_t from, uint8_t to) {
    int8_t fromx = (int8_t)index_to_x(from);
    int8_t fromy = (int8_t)index_to_y(from);
    int8_t tox = (int8_t)index_to_x(to);
    int8_t toy = (int8_t)index_to_y(to);

    uint8_t take_piece = xy_to_index(tox + (fromx < tox ? -1 : 1), toy + (fromy < toy ? -1 : 1));
    if(!enemy_at(take_piece)) return false;

    board_state[take_piece] = 0;

    if (black_turn) {
        white_piece_count--;
    } else {
        black_piece_count--;
    }
    return true;
}

void GameState::play_move(uint16_t move) {
    turns_until_draw--;
    piece_to_move = 0xFF;
    uint8_t from = get_from(move);
    uint8_t to = get_to(move);
    if(man_at(to)) turns_until_draw = TURNS_UNTIL_DRAW;

    if (black_turn) {
        if (to / 4 == 7) {
            board_state[to] = 4;
        } else {
            board_state[to] = board_state[from];
        }
    } else {
        if (to / 4 == 0) {
            board_state[to] = 3;
        } else {
            board_state[to] = board_state[from];
        }
    }
    board_state[from] = 0;

    if (take_piece(from, to)) {
        turns_until_draw = TURNS_UNTIL_DRAW;
        can_take = false;
        valid_move_count = 0;
        if (man_at(to)) {
            add_man_take_moves(to);
        } else {
            add_king_valid_moves(to);
        }
        if (can_take) {
            black_turn = !black_turn;
            piece_to_move = to;
        } else {
            valid_move_count = 0xFF;
        }
    } else {
        valid_move_count = 0xFF;
    }

    black_turn = !black_turn;
}

uint16_t GameState::create_move(uint8_t from, uint8_t to) {
    return (from << 8) + to;
}

uint8_t GameState::get_from(uint16_t move) {
    return move >> 8;
}

uint8_t GameState::get_to(uint16_t move) {
    return move & 0xff;
}

uint8_t GameState::get_pos_nw(uint8_t pos) {
    return pos / 4 % 2 ? (pos % 4 == 0 ? 0xFF : pos - 5) : pos - 4;
}

uint8_t GameState::get_pos_ne(uint8_t pos) {
    return pos / 4 % 2 ? pos - 4 : (pos % 4 == 3 ? 0xFF : pos - 3);
}

uint8_t GameState::get_pos_sw(uint8_t pos) {
    return pos / 4 % 2 ? (pos % 4 == 0 ? 0xFF : pos + 3) : pos + 4;
}

uint8_t GameState::get_pos_se(uint8_t pos) {
    return pos / 4 % 2 ? pos + 4 : (pos % 4 == 3 ? 0xFF : pos + 5);
}

int GameState::index_to_x(int index) {
    return (index % 4) * 2 + (index % 8 < 4);
}

int GameState::index_to_y(int index) {
    return index / 4;
}

int GameState::xy_to_index(int x, int y) {
    return x / 2 + y * 4;
}

void GameState::calculate_game_state() {
    valid_move_count = 0;
    if(white_piece_count == 0) {
        result = 0;
        finished = true;
        return;
    }

    if(black_piece_count == 0) {
        result = 2;
        finished = true;
        return;
    }

    if(turns_until_draw == 0) {
        finished = true;
        return;
    }

    if (piece_to_move != 0xFF) {
        if (man_at(piece_to_move)) {
            add_man_take_moves(piece_to_move);
        } else {
            add_king_valid_moves(piece_to_move);
        }
        if(valid_move_count == 0) finished = true;
        if(valid_move_count > MAX_VALID_MOVES) std::cerr << "Too many moves." << std::endl;
        return;
    }
    can_take = false;
    for (uint8_t i = 0; i < 32; i++) {
        if (empty_at(i) || enemy_at(i)) continue;

        if (man_at(i)) {
            add_man_valid_moves(i);
        } else {
            add_king_valid_moves(i);
        }
    }
    if(valid_move_count == 0) {
        finished = true;
        result = black_turn ? 2 : 0;
    }
    if(valid_move_count > MAX_VALID_MOVES) std::cerr << "Too many moves." << std::endl;
}

bool GameState::move_valid(uint16_t move) {
    if (valid_move_count == 0xFF) calculate_game_state();
    for (int i = 0; i < valid_move_count; i++) {
        if (valid_moves[i] == move) return true;
    }
    return false;
}

void GameState::add_man_non_take_moves(uint8_t pos) {
    uint8_t se = get_pos_se(pos);
    uint8_t sw = get_pos_sw(pos);
    uint8_t ne = get_pos_ne(pos);
    uint8_t nw = get_pos_nw(pos);

    if (black_turn) {
        if (se < 32 && empty_at(se)) valid_moves[valid_move_count++] = create_move(pos, se);
        if (sw < 32 && empty_at(sw)) valid_moves[valid_move_count++] = create_move(pos, sw);
    } else {
        if (ne < 32 && empty_at(ne)) valid_moves[valid_move_count++] = create_move(pos, ne);
        if (nw < 32 && empty_at(nw)) valid_moves[valid_move_count++] = create_move(pos, nw);
    }
}

void GameState::add_man_take_moves(uint8_t pos) {
    uint8_t se = get_pos_se(pos);
    uint8_t se2 = get_pos_se(se);
    uint8_t sw = get_pos_sw(pos);
    uint8_t sw2 = get_pos_sw(sw);
    uint8_t ne = get_pos_ne(pos);
    uint8_t ne2 = get_pos_ne(ne);
    uint8_t nw = get_pos_nw(pos);
    uint8_t nw2 = get_pos_nw(nw);

    if (se < 32 && enemy_at(se) && se2 < 32 && empty_at(se2)) {
        if (!can_take) {
            can_take = true;
            valid_move_count = 0;
        }
        valid_moves[valid_move_count++] = create_move(pos, se2);
    }
    if (sw < 32 && enemy_at(sw) && sw2 < 32 && empty_at(sw2)) {
        if (!can_take) {
            can_take = true;
            valid_move_count = 0;
        }
        valid_moves[valid_move_count++] = create_move(pos, sw2);
    }
    if (ne < 32 && enemy_at(ne) && ne2 < 32 && empty_at(ne2)) {
        if (!can_take) {
            can_take = true;
            valid_move_count = 0;
        }
        valid_moves[valid_move_count++] = create_move(pos, ne2);
    }
    if (nw < 32 && enemy_at(nw) && nw2 < 32 && empty_at(nw2)) {
        if (!can_take) {
            can_take = true;
            valid_move_count = 0;
        }
        valid_moves[valid_move_count++] = create_move(pos, nw2);
    }
}

void GameState::add_man_valid_moves(uint8_t pos) {
    add_man_take_moves(pos);
    if (can_take) return;
    add_man_non_take_moves(pos);
}

bool GameState::enemy_at(uint8_t pos) const {
    return !empty_at(pos) && (board_state[pos] % 2 == black_turn);
}

bool GameState::ally_at(uint8_t pos) const {
    return !empty_at(pos) && (board_state[pos] % 2 != black_turn);
}

bool GameState::empty_at(uint8_t pos) const {
    return board_state[pos] == 0;
}

bool GameState::man_at(uint8_t pos) const {
    return !empty_at(pos) && (board_state[pos] <= 2);
}

bool inline GameState::king_at(uint8_t pos) const {
    return !empty_at(pos) && (board_state[pos] > 2);
}

void GameState::add_king_valid_moves(uint8_t pos) {
    uint8_t p;
    uint8_t p2;
    for (p = get_pos_se(pos); p < 32 && empty_at(p); p = get_pos_se(p)) {
        if (can_take) continue;
        valid_moves[valid_move_count++] = create_move(pos, p);
    }
    if (p < 32 && enemy_at(p)) {
        p2 = get_pos_se(p);
        if (p2 < 32 && empty_at(p2)) {
            if (!can_take) {
                valid_move_count = 0;
                can_take = true;
            }
            valid_moves[valid_move_count++] = create_move(pos, p2);
        }
    }
    for (p = get_pos_sw(pos); p < 32 && empty_at(p); p = get_pos_sw(p)) {
        if (can_take) continue;
        valid_moves[valid_move_count++] = create_move(pos, p);
    }
    if (p < 32 && enemy_at(p)) {
        p2 = get_pos_sw(p);
        if (p2 < 32 && empty_at(p2)) {
            if (!can_take) {
                valid_move_count = 0;
                can_take = true;
            }
            valid_moves[valid_move_count++] = create_move(pos, p2);
        }
    }
    for (p = get_pos_ne(pos); p < 32 && empty_at(p); p = get_pos_ne(p)) {
        if (can_take) continue;
        valid_moves[valid_move_count++] = create_move(pos, p);
    }
    if (p < 32 && enemy_at(p)) {
        p2 = get_pos_ne(p);
        if (p2 < 32 && empty_at(p2)) {
            if (!can_take) {
                valid_move_count = 0;
                can_take = true;
            }
            valid_moves[valid_move_count++] = create_move(pos, p2);
        }
    }
    for (p = get_pos_nw(pos); p < 32 && empty_at(p); p = get_pos_nw(p)) {
        if (can_take) continue;
        valid_moves[valid_move_count++] = create_move(pos, p);
    }
    if (p < 32 && enemy_at(p)) {
        p2 = get_pos_nw(p);
        if (p2 < 32 && empty_at(p2)) {
            if (!can_take) {
                valid_move_count = 0;
                can_take = true;
            }
            valid_moves[valid_move_count++] = create_move(pos, p2);
        }
    }
}

int8_t GameState::simulate_game() const {
    GameState copy(*this);
    while(!copy.finished) {
        uint16_t move = copy.valid_moves[rand() % copy.valid_move_count];
        copy.play_move(move);
        copy.calculate_game_state();
    }
    return copy.result;
}
