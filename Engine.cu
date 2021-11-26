#include <iostream>
#include <chrono>

#include "Engine.cuh"

#include "Logger.cuh"

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
    auto start = std::chrono::high_resolution_clock::now();
    perform_monte_carlo_timed(initial_time);
    int time = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::high_resolution_clock::now() - start)
                   .count();
    total_time += time;
}

const uint8_t *Engine::get_board_state() const
{
    return node->game_state.board_state;
}

Engine::Engine(void (*simulation_step)(Node *, int), int game_count, int initial_time) : node(new Node(simulation_step, game_count)),
                                                                                         initial_time(initial_time)
{
    first_node = node;

    auto start = std::chrono::high_resolution_clock::now();

    steps_per_move = perform_monte_carlo_timed(initial_time);

    int time = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::high_resolution_clock::now() - start)
                   .count();
    total_time += time;
}

Engine::~Engine()
{
    delete first_node;
}

int Engine::perform_monte_carlo_timed(int time)
{
    int step_count = 0;
    int min_steps =  node->game_state.valid_move_count - node->child_count;
    auto start = std::chrono::high_resolution_clock::now();
    int dt;
    while ((dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - start)
                     .count()) <= time ||
           step_count < min_steps)
    {
        node->step();
        step_count++;
    }
    if (step_count == min_steps)
        std::cerr << "Decision too long. (" << dt << "ms)\n";
    return step_count;
}

void Engine::perform_monte_carlo_steps(int steps)
{
    for (int i = 0; i < steps || node->game_state.valid_move_count > node->child_count; i++)
    {
        node->step();
    }
}

Engine::Engine(const Engine &engine) : Engine(engine.node->simulation_step, engine.node->game_count, engine.initial_time) {}
