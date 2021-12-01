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

float play_games(const Engine &e1, const Engine &e2, int game_count = 200)
{
    int points = 0;

    auto start = std::chrono::high_resolution_clock::now();
    points += rand() % 2 ? 2 - play_one_game(e1, e2) : play_one_game(e2, e1);
    int time = std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::high_resolution_clock::now() - start)
                   .count();
    time *= game_count - 1;
    int hours = time / 3600;
    int minutes = (time - hours * 3600) / 60;
    int seconds = (time - hours * 3600 - minutes * 60);
    std::cout << "Expected time: " << hours << "h " << minutes << "m " << seconds << "s\n";

    for (int i = 1; i < game_count; i++)
    {
        points += rand() % 2 ? 2 - play_one_game(e1, e2) : play_one_game(e2, e1);
        if (i % 5 == 4)
            std::cout << '.' << std::flush;
    }
    std::cout << std::endl;
    return 0.5f * points / game_count;
}

void test_method(void (*method)(GameState, uint8_t *, int, int), int game_count, GameState gs = GameState())
{
    uint8_t *res;
    cudaMalloc(&res, game_count);

    auto start = std::chrono::high_resolution_clock::now();

    cudaMemset(res, 0, game_count);
    cudaDeviceSynchronize();

    auto time = std::chrono::high_resolution_clock::now();
    auto time_memset = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           time - start)
                           .count();
    start = time;

    method<<<(game_count - 1) / 1024 + 1, 1024>>>(gs, res, game_count, std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count());
    cudaDeviceSynchronize();

    time = std::chrono::high_resolution_clock::now();
    auto time_method = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           time - start)
                           .count();
    start = time;

    int result = thrust::reduce(thrust::device, res, res + game_count, 0);
    cudaDeviceSynchronize();

    time = std::chrono::high_resolution_clock::now();
    auto time_reduce = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           time - start)
                           .count();
    std::string method_name;
    if (method == run_simulation_step_0)
    {
        method_name = "run_simulation_step_0";
    }
    else if (method == run_simulation_step_1)
    {
        method_name = "run_simulation_step_1";
    }
    Logger::save_time_record(method_name, game_count, time_memset, time_method, time_reduce);

    cudaFree(res);
}

void run_test_winrate()
{
    for (int time = 100; time <= 400; time *= 2)
    {
        // double winrate = play_games(Engine(Node::simulation_step_gpu<run_simulation_step_1>, 4096, time),
        //                             Engine(Node::simulation_step_gpu<run_simulation_step_0>, 4096, time));
        // Logger::save_winrate_record("run_simulation_step_1", 4096, "run_simulation_step_0", 4096, time, winrate);

        // winrate = play_games(Engine(Node::simulation_step_gpu<run_simulation_step_1>, 4096, time),
        //                             Engine(Node::simulation_step_cpu, 1, time));
        // Logger::save_winrate_record("run_simulation_step_1", 4096, "cpu", 1, time, winrate);

        double winrate = play_games(Engine(Node::simulation_step_gpu<run_simulation_step_0>, 4096, time),
                                    Engine(Node::simulation_step_cpu, 1, time));
        Logger::save_winrate_record("run_simulation_step_0", 4096, "cpu", 1, time, winrate);

    }
}

void run_test_performance()
{
    for (int j = 0; j < 20; j++)
    {
        GameState gs = GameState::get_random_state();
        for (int game_count = 1024; game_count <= 1024 * 1024; game_count *= 2)
        {
            if (j % 2)
            {
                test_method(run_simulation_step_1, game_count, gs);
                test_method(run_simulation_step_0, game_count, gs);
            }
            else
            {
                test_method(run_simulation_step_0, game_count, gs);
                test_method(run_simulation_step_1, game_count, gs);
            }

            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < game_count; i++)
            {
                gs.simulate_game();
            }
            auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::high_resolution_clock::now() - start)
                            .count();
            Logger::save_time_record("cpu", game_count, 0, time, 0);
        }
    }
}

int main(int argc, char **argv)
{
    if (argc >= 3)
    {
        long seed = time(nullptr);
        srand(seed);

        Window window(512, 512);
        window.set_players(argv[1], argv[2]);

        window.run();

        return EXIT_SUCCESS;
    }

    Logger::init("log_time", "log_winrate", "v3");
    run_test_performance();
    run_test_winrate();
    return EXIT_SUCCESS;
}
