#include <iostream>
#include <chrono>
#include "Engine.cuh"

uint16_t Engine::get_move() {
    Node *best = node->children[node->best_child_index()];
    return best->move;
}

void Engine::play_move(uint16_t move) {
    int chosen_node = node->get_move_index(move);
    for (int i = 0; i < node->game_state.valid_move_count; i++) {
        if (i == chosen_node) continue;
        delete node->children[i];
        node->children[i] = nullptr;
    }
    node = node->children[chosen_node];
    perform_monte_carlo();
}

const uint8_t *Engine::get_board_state() const {
    return node->game_state.board_state;
}

Engine::Engine(void(*simulation_step)(Node *), int time_per_move) : node(new Node(simulation_step)),
                                                                    time_per_move(time_per_move) {
    first_node = node;
    perform_monte_carlo();
}

Engine::~Engine() {
    delete first_node;
}

int Engine::perform_monte_carlo() {
    int step_count = 0;
    auto start = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count() <= time_per_move) {
        node->step();
        step_count++;
    }
    std::cout << step_count << std::endl;
    return step_count;
}

Engine::Engine(const Engine& engine) : Engine(engine.node->simulation_step, engine.time_per_move) {}
