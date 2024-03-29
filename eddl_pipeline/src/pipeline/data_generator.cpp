#include "data_generator.hpp"

#include <iostream>

DataGenerator::DataGenerator(ecvl::DLDataset *source, int batch_size,
                             std::vector<int> input_shape,
                             std::vector<int> output_shape, int n_producers)
    : source_(source), batch_size_(batch_size), input_shape_(input_shape),
      output_shape_(output_shape), n_producers_(n_producers) {
  const int num_samples = ecvl::vsize(source_->GetSplit());
  num_batches_ = num_samples / batch_size_;

  producers_ = new std::thread[n_producers_];

  active_ = false;
  batch_index_ = 0;
  this->active_producers_ = 0;
}

DataGenerator::~DataGenerator() {
  if (active_)
    Stop();

  // this loop can be executed with no control of mutex if producer stopped
  while (!fifo_.empty()) {
    TensorPair *_temp = fifo_.front();
    fifo_.pop();

    delete _temp->x_;
    delete _temp->y_;
    delete _temp;
  }

  delete[] producers_;
}

void DataGenerator::Start() {
  if (active_) {
    std::cerr << "FATAL ERROR: trying to start the producer threads when "
                 "they are already running!"
              << std::endl;
    std::abort();
  }

  batch_index_ = 0;
  active_ = true;
  this->active_producers_ = 0;
  for (int i = 0; i < n_producers_; i++) {
    producers_[i] = std::thread(&DataGenerator::ThreadProducer, this);
  }
}

void DataGenerator::Stop() {
  if (!active_) {
    std::cerr << "FATAL ERROR: trying to stop the producer threads when "
                 "they are stopped!"
              << std::endl;
    std::abort();
  }

  active_ = false;
  for (int i = 0; i < n_producers_; i++)
    producers_[i].join();
  this->active_producers_ = 0;
}

void DataGenerator::increase_producers()
{
    // Critical region starts
    std::unique_lock<std::mutex> lck(mutex_batch_index_);

    this->active_producers_++;
    // Critical region ends
}
void DataGenerator::decrease_producers()
{
    // Critical region starts
    std::unique_lock<std::mutex> lck(mutex_batch_index_);

    this->active_producers_--;
    // Critical region ends
}

void DataGenerator::ThreadProducer() {

  this->increase_producers();
  std::queue<TensorPair *> my_cache;

  while (active_ && batch_index_ < num_batches_) {
    int j = -1;

    { // Critical region starts
      std::unique_lock<std::mutex> lck(mutex_batch_index_);

      j = batch_index_++;
    } // Critical region ends

    if (j >= num_batches_)
      break;

    // Creating new tensors for every batch can generate overload, let us check
    // now this in the future
    Tensor *x = new Tensor({batch_size_, source_->n_channels_, input_shape_[0], input_shape_[1]});
    Tensor *y;
    if (source_->classes_.empty())
      y = new Tensor({batch_size_, source_->n_channels_gt_, input_shape_[0], input_shape_[1]});
    else
      y = new Tensor({batch_size_, ecvl::vsize(source_->classes_)});

    // Load a batch
    source_->LoadBatch(x, y);

    TensorPair *p = new TensorPair(x, y);

    my_cache.push(p);

    if (fifo_.size() < 20 || my_cache.size() > 5) {
      { // Critical region starts
        std::unique_lock<std::mutex> lck(mutex_fifo_);

        while (!my_cache.empty()) {
          fifo_.push(my_cache.front());
          my_cache.pop();
        }
      } // Critical region ends
      cond_var_fifo_.notify_one();
    }

    // Pending to be adjusted
    while (fifo_.size() > 100) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }

  if (!my_cache.empty()) {
    { // Critical region starts
      std::unique_lock<std::mutex> lck(mutex_fifo_);
      while (!my_cache.empty()) {
        fifo_.push(my_cache.front());
        my_cache.pop();
      }
    } // Critical region ends
    cond_var_fifo_.notify_one();
  }

  this->decrease_producers();
}

bool DataGenerator::HasNext() {
  //return batch_index_ < num_batches_ || !fifo_.empty();
  return batch_index_ < num_batches_ || !fifo_.empty() || this->active_producers_ > 0;
}

size_t DataGenerator::Size() { return fifo_.size(); }

bool DataGenerator::PopBatch(Tensor *&x, Tensor *&y) {
  TensorPair *_temp;
  { // Critical region begins
    std::unique_lock<std::mutex> lck(mutex_fifo_);

    if (fifo_.empty())
      cond_var_fifo_.wait(lck);

    if (fifo_.empty())
      return false;

    _temp = fifo_.front();
    fifo_.pop();
  } // Critical region ends

  x = _temp->x_;
  y = _temp->y_;

  delete _temp;

  return true;
}
