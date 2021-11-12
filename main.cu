#include <iostream>
#include "Window.cuh"
#include "Logger.cuh"

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

void run_tests()
{
    for (int i = 0; i < 100; i++)
    {
        for (int time = 10; time <= 1000; time *= 10)
        {
            for (int game_count = 100; game_count <= 1000000; game_count *= 10)
            {
                Engine(Node::simulation_step_gpu<run_simulation_step_0>, game_count, time);
                Engine(Node::simulation_step_cpu, 1, time);
            }
        }
    }
}

int main()
{
    Logger::init("log", "2");
    long seed = time(nullptr);
    srand(seed);

    run_tests();

    // std::cout << compare(Node::simulation_step_gpu<run_simulation_step_0>, 1024, Node::simulation_step_cpu, 1, 10) << std::endl;
    // std::cout << compare(Node::simulation_step_cpu, 1, Node::simulation_step_cpu, 10, 10) << std::endl;

    //    std::cout << "Seed: "  << seed << std::endl;
    //    Window window(512, 512);
    //    window.run();
}
