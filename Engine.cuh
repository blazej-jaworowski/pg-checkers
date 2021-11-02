#ifndef PG_MONTE_CARLO_CHECKERS_ENGINE_CUH
#define PG_MONTE_CARLO_CHECKERS_ENGINE_CUH

#include <cstdint>

class Engine {
    uint8_t board_state[32]{};
public:
    Engine();
    explicit Engine(const uint8_t * board_state);
    uint16_t get_move();
    bool play_move(uint16_t move);
    const uint8_t * get_board_state();
    static inline uint16_t create_move(uint8_t from, uint8_t to);
    static inline uint8_t get_from(uint16_t move);
    static inline uint8_t get_to(uint16_t move);
};


#endif
