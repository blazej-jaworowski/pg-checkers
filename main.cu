#include <iostream>
#include "Window.cuh"

int play_one_game(Engine black, Engine white) {
    GameState gs;
    while (!gs.finished) {
        uint16_t move = gs.black_turn ? black.get_move() : white.get_move();
        black.play_move(move);
        white.play_move(move);
        gs.play_move(move);
        gs.calculate_game_state();
    }
    return gs.result;
}

float play_games(const Engine &e1, const Engine &e2, int game_count = 100) {
    int points = 0;
    for (int i = 0; i < game_count; i++) {
//        std::cout << i << std::endl;
        points += rand() % 2 ? 2 - play_one_game(e1, e2) : play_one_game(e2, e1);
    }
    return 0.5f * points / game_count;
}

float compare(void(*method)(Node*), void(*base_method)(Node*), int time) {
    Engine e(method, time);
    Engine base(base_method, time);
    float result;
    do {
        result = play_games(e, base);
        result = result / (1 - result);
        e.time_per_move /= result;
    } while (abs(result) > 0.1);
}

int main() {
    long seed = time(nullptr);
    srand(seed);

    for(int i = 0; i < 6; i++) {
        std::cout << 100 + i * 50 << ": ";
        for(int j = 0; j < 10; j++) {
            std::cout << play_games(Engine(Node::simulation_step_cpu, 100),
                                    Engine(Node::simulation_step_cpu, 100), 100 + i * 50) << " ";
        }
        std::cout << std::endl;
    }

//    std::cout << "Seed: "  << seed << std::endl;
//    Window window(512, 512);
//    window.run();
}
