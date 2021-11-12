#ifndef PG_CHECKERS_LOGGER_CUH
#define PG_CHECKERS_LOGGER_CUH

#include <string>
#include <fstream>

class Logger
{
private:
    static Logger logger;

    std::string version;
    std::ofstream out;
    bool enabled = false;

    void open(std::string filename);
public:
    ~Logger();
    static void init(std::string filename, std::string version);
    static void save_record(std::string method, int game_count, int time, int steps);
};

#endif