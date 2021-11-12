#ifndef PG_MONTE_CARLO_CHECKERS_GAMESTATE_CUH
#define PG_MONTE_CARLO_CHECKERS_GAMESTATE_CUH

#include <cstdint>
#include <thrust/random.h>

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

    __host__ __device__ static uint16_t create_move(uint8_t from, uint8_t to);
    __host__ __device__ static uint8_t get_from(uint16_t move);
    __host__ __device__ static uint8_t get_to(uint16_t move);
    __host__ __device__ static uint8_t get_pos_nw(uint8_t pos);
    __host__ __device__ static uint8_t get_pos_ne(uint8_t pos);
    __host__ __device__ static uint8_t get_pos_sw(uint8_t pos);
    __host__ __device__ static uint8_t get_pos_se(uint8_t pos);
    __host__ __device__ static int index_to_x(int index);
    __host__ __device__ static int index_to_y(int index);
    __host__ __device__ static int xy_to_index(int x, int y);

    __host__ __device__ bool ally_at(uint8_t pos) const;
    __host__ __device__ bool enemy_at(uint8_t pos) const;
    __host__ __device__ bool empty_at(uint8_t pos) const;

    __host__ __device__ bool man_at(uint8_t pos) const;
    __host__ __device__ bool king_at(uint8_t pos) const;

    __host__ __device__ bool take_piece(uint8_t from, uint8_t to);
    __host__ __device__ void play_move(uint16_t move);
    __host__ __device__ bool move_valid(uint16_t move);

    __host__ __device__ void add_king_valid_moves(uint8_t pos);
    __host__ __device__ void add_man_valid_moves(uint8_t pos);
    __host__ __device__ void add_man_non_take_moves(uint8_t pos);
    __host__ __device__ void add_man_take_moves(uint8_t pos);
    __host__ __device__ void calculate_game_state();

    __host__ __device__ int8_t simulate_game() const;
    __host__ __device__ int8_t simulate_game(thrust::random::minstd_rand &rng) const;

    __host__ __device__ GameState();
};


#endif