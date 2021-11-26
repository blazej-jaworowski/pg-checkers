#include "Logger.cuh"

Logger Logger::logger;

void Logger::open(std::string filename)
{
    out.open(filename, std::ios::app);
}

Logger::~Logger()
{
    out.close();
}

void Logger::init(std::string filename, std::string version)
{
    logger.enabled = true;
    logger.version = version;
    logger.open(filename);
}

void Logger::save_record(std::string method, int game_count,
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

    logger.out << logger.version << "," << build << "," << method << "," << game_count << "," << time_memset << "," << time_simulation << "," << time_reduce << std::endl;
}