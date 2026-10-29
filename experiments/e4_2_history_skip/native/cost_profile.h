#pragma once
#include <chrono>
#include <array>
namespace diskann {
// Exclusive software-region wall time. Not hardware memory/SSD device latency.
struct E1Cost {
    using Clock=std::chrono::steady_clock;
    std::array<double,5> us{};
    int current=4;
    Clock::time_point last=Clock::now();
    void change(int next) { auto now=Clock::now();us[current]+=std::chrono::duration<double,std::micro>(now-last).count();last=now;current=next; }
};
inline thread_local E1Cost* e1_cost=nullptr;
struct E1Scope {
    E1Cost* p; int old;
    explicit E1Scope(int category):p(e1_cost),old(p?p->current:4){if(p)p->change(category);}
    ~E1Scope(){if(p)p->change(old);}
};
}
