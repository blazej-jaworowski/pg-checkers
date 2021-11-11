#ifndef PG_MONTE_CARLO_CHECKERS_GAMESTATE_CUH
#define PG_MONTE_CARLO_CHECKERS_GAMESTATE_CUH

#include <cstdint>

#define MAX_VALID_MOVES 128
#define TURNS_UNTIL_DRAW 40

class GameState {
public:
    uint8_t board_state[32]{};
    bool black_turn = true;
    uint16_t valid_moves[MAX_VALID_MOVES];
    uint8_t valid_move_count = 0xFF;
    bool can_take = false;

    uint8_t piece_to_move = 0xFF;

    uint8_t black_piece_count;
    uint8_t white_piece_count;
    uint8_t turns_until_draw = TURNS_UNTIL_DRAW;

    bool finished = false;
    int8_t result = 1;

    static inline uint16_t create_move(uint8_t from, uint8_t to);
    static inline uint8_t get_from(uint16_t move);
    static inline uint8_t get_to(uint16_t move);
    static inline uint8_t get_pos_nw(uint8_t pos);
    static inline uint8_t get_pos_ne(uint8_t pos);
    static inline uint8_t get_pos_sw(uint8_t pos);
    static inline uint8_t get_pos_se(uint8_t pos);
    static inline int index_to_x(int index);
    static inline int index_to_y(int index);
    static inline int xy_to_index(int x, int y);

    bool inline ally_at(uint8_t pos) const;
    bool inline enemy_at(uint8_t pos) const;
    bool inline empty_at(uint8_t pos) const;

    bool inline man_at(uint8_t pos) const;
    bool inline king_at(uint8_t pos) const;

    bool take_piece(uint8_t from, uint8_t to);
    void play_move(uint16_t move);
    bool move_valid(uint16_t move);

    void add_king_valid_moves(uint8_t pos);
    void add_man_valid_moves(uint8_t pos);
    void add_man_non_take_moves(uint8_t pos);
    void add_man_take_moves(uint8_t pos);
    void calculate_game_state();

    int8_t simulate_game() const;

    GameState();
};


#endif