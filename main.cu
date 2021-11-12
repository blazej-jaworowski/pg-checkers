#include <iostream>
#include "Window.cuh"

int play_one_game(Engine black, Engine white)
{
    GameState gs;
    while (!gs.finished)
    {
        uint16_t move = gs.black_turn ? black.get_move() : white.get_move();
        black.play_move(move);
        white.play_move(move);
        gs.play_move(move);
        gs.calculate_game_state();
    }
    return gs.result;
}

float play_games(const Engine &e1, const Engine &e2, int game_count = 100)
{
    int points = 0;
    for (int i = 0; i < game_count; i++)
    {
        points += rand() % 2 ? 2 - play_one_game(e1, e2) : play_one_game(e2, e1);
        if (i % 5 == 4)
            std::cout << '.' << std::flush;
    }
    std::cout << std::endl;
    return 0.5f * points / game_count;
}

float compare(void (*method)(Node *, int), int game_count, void (*base_method)(Node *, int), int game_count_base, int time)
{
    Engine e(method, game_count, time);
    Engine base(base_method, game_count_base, time);
    float result;
    do
    {
        result = play_games(e, base);
        std::cout << result << std::endl;
        result = result / (1 - result);
        e.time_per_move /= result;
        std::cout << e.time_per_move << std::endl;
    } while (abs(result - 1) > 0.1);
    return 1.0 * time / e.time_per_move;
}

__global__ void run_simulation_step_0(GameState game_state, uint8_t * results, int game_count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index > game_count) return;

    results[index] = game_state.simulate_game();
}

int main()
{
    long seed = time(nullptr);
    srand(seed);

    std::cout << compare(Node::simulation_step_cpu, 1, Node::simulation_step_gpu<run_simulation_step_0>, 1024, 3) << std::endl;

    // for (int i = 0; i < 7; i++)
    // {
    //     std::cout << 100 + i * 50 << ": ";
    //     for (int j = 0; j < 50; j++)
    //     {
    //         std::cout << play_games(Engine(Node::simulation_step_cpu, 1, 1),
    //                                 Engine(Node::simulation_step_cpu, 1, 1), 100 + i * 50)
    //                   << " " << std::flush;
    //     }
    //     std::cout << std::endl;
    // }

    //    std::cout << "Seed: "  << seed << std::endl;
    //    Window window(512, 512);
    //    window.run();
}
