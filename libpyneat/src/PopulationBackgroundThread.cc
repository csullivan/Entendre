#include "PopulationBackgroundThread.hh"

#include <iostream>

PopulationBackgroundThread::PopulationBackgroundThread(Population pop)
  : current_pop(pop), stop_worker(false), num_generations(0) {

  worker_thread = std::thread([this](){worker_loop();});
}

PopulationBackgroundThread::~PopulationBackgroundThread() {
  {
    std::lock_guard<std::mutex> lock(wake_worker_mutex);
    stop_worker = true;
    wake_worker_cond.notify_one();
  }
  worker_thread.join();
}

void PopulationBackgroundThread::worker_loop() {
  while(true) {
    std::unique_lock<std::mutex> lock(wake_worker_mutex);
    wake_worker_cond.wait(lock, [this](){
        return stop_worker || num_generations;
      } );

    if(stop_worker) {
      break;
    }

    if(num_generations) {
      // For first population, just evaluate.
      // For all others, reproduce, then evaluate the next population.
      if(current_pop.IsEvaluated()) {
        current_pop = current_pop.Reproduce();
      }
      current_pop.Evaluate(fitness);

      // One copy for us to keep, one copy to throw up to the GUI.
      std::lock_guard<std::mutex> lock(output_queue_mutex);
      output_queue.push(std::make_unique<Population>(current_pop));

      num_generations--;
    }
  }
}


void PopulationBackgroundThread::perform_reproduction(
  std::function<double(NeuralNet&)> fitness,
  unsigned int num_generations) {

  std::lock_guard<std::mutex> lock(wake_worker_mutex);
  this->fitness = fitness;
  this->num_generations = num_generations;
  wake_worker_cond.notify_one();
}

std::unique_ptr<Population> PopulationBackgroundThread::get_next_generation() {
  std::lock_guard<std::mutex> lock(output_queue_mutex);
  if(output_queue.size()) {
    auto output = std::move(output_queue.front());
    output_queue.pop();
    return output;
  } else {
    return nullptr;
  }
}
