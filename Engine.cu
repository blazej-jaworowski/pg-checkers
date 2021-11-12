#include <iostream>
#include <chrono>
#include "Engine.cuh"

uint16_t Engine::get_move()
{
    Node *best = node->children[node->best_child_index()];
    return best->move;
}

void Engine::play_move(uint16_t move)
{
    int chosen_node = node->get_move_index(move);
    for (int i = 0; i < node->game_state.valid_move_count; i++)
    {
        if (i == chosen_node)
            continue;
        delete node->children[i];
        node->children[i] = nullptr;
    }
    node = node->children[chosen_node];
    perform_monte_carlo();
}

const uint8_t *Engine::get_board_state() const
{
    return node->game_state.board_state;
}

Engine::Engine(void (*simulation_step)(Node *, int), int game_count, int time_per_move) : node(new Node(simulation_step, game_count)),
                                                                                          time_per_move(time_per_move)
{
    first_node = node;
    perform_monte_carlo();
}

Engine::~Engine()
{
    delete first_node;
}

int Engine::perform_monte_carlo()
{
    int step_count = 0;
    int loop_count = 0;
    auto start = std::chrono::high_resolution_clock::now();
    int dt;
    while ((dt = std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::high_resolution_clock::now() - start)
               .count()) <= time_per_move)
    {
        for (int i = 0; i < node->game_state.valid_move_count; i++)
        {
            node->step();
            step_count++;
        }
        loop_count++;
    }
    if(loop_count == 1) std::cerr << "Decision too long. (" << dt << "ms)\n";
    return step_count;
}

Engine::Engine(const Engine &engine) : Engine(engine.node->simulation_step, engine.node->game_count, engine.time_per_move) {}
