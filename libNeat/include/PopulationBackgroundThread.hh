#ifndef _POPULATIONBACKGROUNDTHREAD_H_
#define _POPULATIONBACKGROUNDTHREAD_H_

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <queue>

#include "Population.hh"

/// Helper class to perform heavy lifting in a background thread.
class PopulationBackgroundThread {
public:
  /// Initializes
  PopulationBackgroundThread(Population pop);
  ~PopulationBackgroundThread();

  /// Request a reproduction to be done
  void perform_reproduction(std::function<double(NeuralNet&)> fitness,
                            unsigned int num_generations);

  /// Returns the next generation, if present,
  /**
     The first output will be the initial population, evaluated with
     the fitness function.  Each subsequent output will be the next
     generation, evaluated with the fitness function.  If there are no
     populations available, nullptr will be returned.
   */
  std::unique_ptr<Population> get_next_generation();

private:
  void worker_loop();

  Population current_pop;
  std::thread worker_thread;

  std::mutex priority_mutex;
  std::condition_variable wake_worker_cond;
  std::mutex wake_worker_mutex;

  bool stop_worker;
  std::function<double(NeuralNet&)> fitness;
  unsigned int num_generations;

  std::mutex output_queue_mutex;
  std::queue<std::unique_ptr<Population> > output_queue;
};

#endif /* _POPULATIONBACKGROUNDTHREAD_H_ */
