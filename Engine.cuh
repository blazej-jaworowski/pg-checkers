#ifndef PG_MONTE_CARLO_CHECKERS_ENGINE_CUH
#define PG_MONTE_CARLO_CHECKERS_ENGINE_CUH

#include <cstdint>
#include "GameState.cuh"
#include "Node.cuh"

class Engine {
private:
    Node * first_node;
public:
    Node * node;
    explicit Engine(void(*simulation_step)(Node*) = Node::simulation_step_cpu, int time_per_move = 100);
    Engine(const Engine&);
    ~Engine();
    uint16_t get_move();

    int time_per_move;

    int perform_monte_carlo();
    void play_move(uint16_t move);
    const uint8_t * get_board_state() const;
};


#endif
