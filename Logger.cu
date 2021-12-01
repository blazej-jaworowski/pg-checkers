#include "Logger.cuh"

Logger Logger::logger;

void Logger::open(std::string filename_time, std::string filename_winrate)
{
    out_time.open(filename_time, std::ios::app);
    out_winrate.open(filename_winrate, std::ios::app);
}

Logger::~Logger()
{
    out_time.close();
}

void Logger::init(std::string filename_time, std::string filename_winrate, std::string version)
{
    logger.enabled = true;
    logger.version = version;
    logger.open(filename_time, filename_winrate);
}

void Logger::save_time_record(std::string method, int game_count,
                              unsigned long long time_memset,
                              unsigned long long time_simulation,
                              unsigned long long time_reduce)
{
    if (!logger.enabled)
        return;

#ifndef NDEBUG
    std::string build = "Debug";
#else
    std::string build = "Release";
#endif

    logger.out_time << logger.version << "," << build << "," << method << "," << game_count << "," << time_memset << "," << time_simulation << "," << time_reduce << std::endl;
}

void Logger::save_winrate_record(std::string method1, int game_count1,
                                 std::string method2, int game_count2,
                                 int time, double winrate)
{
    if (!logger.enabled)
        return;
#ifndef NDEBUG
    std::string build = "Debug";
#else
    std::string build = "Release";
#endif

    logger.out_winrate << logger.version << "," << build << "," << method1 << "," << game_count1 << "," << method2 << "," << game_count2 << "," << time << "," << winrate << std::endl;
}