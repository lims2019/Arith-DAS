#include "VMAC.h"
#include "spdlog/spdlog.h"
#include "verilated.h"
#include <iostream>
#include <memory.h>
#include <memory>
#include <random>

#define INPUT_WIDTH 16
#define TEST_INPUT_WIDTH 16
#define OUTPUT_WIDTH 31

int main()
{
    auto                                    top  = std::make_shared<VMAC>();
    bool                                    flag = true;
    int                                     cnt  = 0;
    std::random_device                      rd;
    std::mt19937                            gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, ((( long long )1) << TEST_INPUT_WIDTH) - 1);
    for (long long i = 0; i < 1e6; i += 1)
    {
        cnt += 1;
        if (cnt % 100 == 0)
        {
            spdlog::info("testing {}", cnt);
        }
        unsigned long long a = dis(gen);
        unsigned long long b = dis(gen);
        unsigned long long c = dis(gen);
        top->a               = a;
        top->b               = b;
        top->c               = c;
        top->eval();

        unsigned long long out          = top->out;
        unsigned long long ground_truth = ((a * b + c) & ((( long )1 << OUTPUT_WIDTH) - 1));
        if (out != ground_truth)
        {
            flag = false;
            spdlog::error("a = {}, b = {}, output out is {}, true value is {}", a, b, out, ground_truth);
        }
        else
        {
            // Wow! The output is correct!
        }
    }

    if (flag)
    {
        spdlog::info("All {} tests passed", cnt);
        spdlog::info("Congratulations!")
    }
    return 0;
}
