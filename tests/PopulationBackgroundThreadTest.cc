#include <chrono>
#include <mutex>
#include <thread>

#include <gtest/gtest.h>

#include "Population.hh"
#include "PopulationBackgroundThread.hh"

TEST(PopulationBackgroundThread, RunNonBlocking) {
  auto adam = Genome()
    .AddNode(NodeType::Bias)
    .AddNode(NodeType::Input)
    .AddNode(NodeType::Hidden)
    .AddNode(NodeType::Output)
    .AddConnection(0,3,true,1.)
    .AddConnection(1,3,true,1.)
    .AddConnection(1,2,true,1.)
    .AddConnection(2,3,true,1.);

  Population pop(adam,
                 std::make_shared<RNG_MersenneTwister>(),
                 std::make_shared<Probabilities>());

  PopulationBackgroundThread bg_thread(std::move(pop));

  std::mutex mutex;
  std::unique_lock<std::mutex> lock(mutex);

  std::vector<_float_> input_vals = {1};
  bg_thread.perform_reproduction(
    // fitness lambda function
    [input_vals, &mutex](NeuralNet& net) {
      std::lock_guard<std::mutex> lock(mutex);
      auto outputs = net.evaluate(input_vals);
      return 1.0;
    },
    2
  );

  // Hasn't been allowed to start, so this must be null.
  auto null_gen = bg_thread.get_next_generation();
  EXPECT_EQ(null_gen, nullptr);

  // Let the worker thread continue
  lock.unlock();

  // There's got to be a better way to test this one.  I want to test
  // that there are eventually two generations returned.
  int num_received = 0;
  for(int i=0; i<100; i++) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto next_gen = bg_thread.get_next_generation();
    if(next_gen) {
      num_received++;
      if(num_received == 2) {
        break;
      }
    }
  }
  EXPECT_EQ(num_received, 2);
}
