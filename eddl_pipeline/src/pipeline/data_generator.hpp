#ifndef _DATAGENERATOR_H_
#define _DATAGENERATOR_H_

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

#include <ecvl/support_eddl.h>

class DataGenerator {
  class TensorPair {
  public:
    TensorPair(Tensor *&x, Tensor *&y) : x_(x), y_(y) {}

    Tensor *x_;
    Tensor *y_;
  };

private:
  ecvl::DLDataset *source_;
  int batch_size_;
  std::vector<int> input_shape_;
  std::vector<int> output_shape_;
  int n_producers_;

  std::queue<TensorPair *> fifo_;
  std::mutex mutex_fifo_;
  std::mutex mutex_batch_index_;
  std::condition_variable cond_var_fifo_;
  std::thread *producers_;
  int batch_index_;

  bool active_;
  int num_batches_;

public:
  DataGenerator(ecvl::DLDataset *source, int batch_size,
                std::vector<int> input_shape, std::vector<int> output_shape,
                int n_producers = 5);
  ~DataGenerator();

  void Start();
  void Stop();
  void ThreadProducer();
  bool HasNext();
  size_t Size();
  bool PopBatch(Tensor *&x, Tensor *&y);
};

#endif
