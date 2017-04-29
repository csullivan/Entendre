#ifndef TIMER_H
#define TIMER_H
#include <functional>
#include <chrono>

using namespace std;

struct Timer {
        Timer(function<void(long long)> callback)
                : callback(callback)
                , t0(chrono::high_resolution_clock::now()) { ; }
        ~Timer(void) {
                auto t1 = chrono::high_resolution_clock::now();
                long long elapsed = chrono::duration_cast<chrono::nanoseconds>(t1-t0).count();
                callback(elapsed);
        }
        function<void(long long)> callback;
        chrono::high_resolution_clock::time_point t0;

};

#endif
