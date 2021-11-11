#ifndef PG_MONTE_CARLO_CHECKERS_GAMESTATE_H
#define PG_MONTE_CARLO_CHECKERS_GAMESTATE_H


class GameState {
public:
    uint8_t board_state[32]{};

    static inline uint16_t create_move(uint8_t from, uint8_t to);
    static inline uint8_t get_from(uint16_t move);
    static inline uint8_t get_to(uint16_t move);
    bool play_move(uint16_t move);
    static void get_possible_moves(uint16_t * move_buffer) const;
    GameState();
};


#endif