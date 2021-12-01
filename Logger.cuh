#ifndef PG_CHECKERS_LOGGER_CUH
#define PG_CHECKERS_LOGGER_CUH

#include <string>
#include <fstream>

class Logger
{
private:
    static Logger logger;

    std::string version;
    std::ofstream out_time;
    std::ofstream out_winrate;
    bool enabled = false;

    void open(std::string filename_time, std::string filename_winrate);

public:
    ~Logger();
    static void init(std::string filename_time, std::string filename_winrate, std::string version);
    static void save_time_record(std::string method, int game_count,
                            unsigned long long time_memset,
                            unsigned long long time_simulation,
                            unsigned long long time_reduce);
    static void save_winrate_record(std::string method1, int game_count1,
                                    std::string method2, int game_count2,
                                    int time, double winrate);
};

#endif