#ifndef PG_MONTE_CARLO_CHECKERS_NODE_CUH
#define PG_MONTE_CARLO_CHECKERS_NODE_CUH

#include "GameState.cuh"

#include <thrust/reduce.h>

class Node
{
public:
    GameState game_state;
    uint16_t move = 0xFFFF;
    bool leaf = true;

    int points = 0;
    int total_points = 0;

    Node *parent;
    Node *children[MAX_VALID_MOVES];
    int child_count = 0;

    void (*simulation_step)(Node *, int);
    int game_count;

    uint8_t * game_result_buffer;

    static void simulation_step_cpu(Node *node, int game_count);

    template <void (*F)(GameState, uint8_t *, int)>
    static void simulation_step_gpu(Node *node, int game_count)
    {
        F<<<game_count / 1024 + 1, 1024>>>(node->game_state, node->game_result_buffer, game_count);
        int result = thrust::reduce(thrust::device, node->game_result_buffer, node->game_result_buffer + game_count, 0);
        node->propagate_result(result, game_count * 2);
    }

    void propagate_result(int white_points, int max_points);
    void expand();
    void step();
    Node *choose_child();
    int best_child_index();

    explicit Node(void (*simulation_step)(Node *, int), int game_count);

    Node(Node *parent, uint16_t move);

    ~Node();

    int get_move_index(uint16_t child_move);
};

__global__ void run_simulation_step_0(GameState game_state, uint8_t *results, int game_count);

#endif
