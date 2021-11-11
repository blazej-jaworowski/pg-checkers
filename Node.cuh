#ifndef PG_MONTE_CARLO_CHECKERS_NODE_CUH
#define PG_MONTE_CARLO_CHECKERS_NODE_CUH

#include "GameState.cuh"

class Node {
public:
    GameState game_state;
    uint16_t move = 0xFFFF;
    bool leaf = true;

    int points = 0;
    int total_points = 0;

    Node * parent;
    Node * children[MAX_VALID_MOVES];
    int child_count = 0;

    void (*simulation_step)(Node*);

    static void simulation_step_cpu(Node * node);

    void propagate_result(int white_points, int max_points);
    void expand();
    void step();
    Node * choose_child();
    int best_child_index();

    explicit Node(void(*simulation_step)(Node*));

    Node(Node* parent, uint16_t move);

    ~Node();

    int get_move_index(uint16_t child_move);
};


#endif
