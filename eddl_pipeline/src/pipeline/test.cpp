#include "test.hpp"
#include <chrono>
#include <iomanip>


#include "data_generator.hpp"

TestResults test(ecvl::DLDataset &dataset, Net *model, Arguments &args) {
    int id = get_id_distributed();

    if (id == 0) {
        // Get test split info
        dataset.SetSplit(ecvl::SplitType::test);
        //const int n_te_samples = dataset.GetSplit().size();
        const int n_te_batches = dataset.GetNumBatches(ecvl::SplitType::test);

        // Reset batch counter
        dataset.ResetAllBatches();
        //dataset.ResetBatch(ecvl::SplitType::test);

        // Reset the accumulated loss value
        eddl::reset_loss(model);

        // Auxiliary variables to store the results
        float loss, acc;

        // Validation phase
        float load_time = 0.f;
        float eval_time = 0.f;
        float misc_time = 0.f;
        const auto test_start = std::chrono::high_resolution_clock::now();
        dataset.Start();
        for (int b = 1; b <= n_te_batches; ++b) {
            // Load data
            const auto load_start = std::chrono::high_resolution_clock::now();
            auto [samples, x, y] = dataset.GetBatch();
            const auto load_end = std::chrono::high_resolution_clock::now();
            load_time += std::chrono::duration_cast<std::chrono::microseconds>(load_end - load_start).count();

            // We make sure that the shape of the model is the correct one
            // This may change for the last batch
            model->resize(x.get()->shape[0]);

            // Perform evaluation
            const auto eval_start = std::chrono::high_resolution_clock::now();
            if (args.classifier_output == "sigmoid" && dataset.classes_.size() == 2) {
                Tensor *aux_y = y.get()->select({":", "1"});
                aux_y->reshape_({-1});
                eddl::eval_batch(model,{x.get()},
                {
                    aux_y
                });
                delete aux_y;
            } else {
                eddl::eval_batch(model,{x.get()},
                {
                    y.get()
                });
            }
            const auto eval_end = std::chrono::high_resolution_clock::now();
            eval_time += std::chrono::duration_cast<std::chrono::microseconds>(eval_end - eval_start).count();

            // Get the current losses and metrics
            loss = eddl::get_losses(model)[0];
            acc = eddl::get_metrics(model)[0];

            //if (id == 0) {
            // Show current stats
            std::cout << "[SEQ] Proc " << id << " - ";
            ;
            std::cout << "Testing: Batch " << b << "/" << n_te_batches << ": ";
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "Metrics[ loss=" << loss << ", acc=" << acc << " ]";
            std::cout << " - Timers[ ";
            std::cout << "avg_load_batch=" << (load_time / b) * 1e-6 << "s";
            std::cout << ", avg_eval_batch=" << (eval_time / b) * 1e-6 << "s ]";
            std::cout << " - DataGenerator[ |fifo| = " << dataset.GetQueueSize() << " ]";
            std::cout << std::endl;
            //}
        }
        int b = n_te_batches;
        // Show current stats
        printf("FINAL: Metrics[ loss= %.4f acc= %.4f ]\n", loss, acc);
       
        dataset.Stop();
       
        const auto test_end = std::chrono::high_resolution_clock::now();
        const float test_time = std::chrono::duration_cast<std::chrono::microseconds>(test_end - test_start).count();
        std::cout << "Test time elapsed = " << test_time * 1e-6 << "s\n\n";

        // Show the epoch results for each split
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Test results: loss=" << loss << " - acc=" << acc << "\n\n";

        return TestResults(loss, acc);
    }
}

TestResults test_distr(ecvl::DLDataset &dataset, Net *model, Arguments &args, bool distr_dataset) {
    int id = get_id_distributed();
    // Get test split info
    dataset.SetSplit(ecvl::SplitType::test);
    const int n_te_samples = dataset.GetSplit().size();
    //const int n_te_batches = dataset.GetNumBatches(ecvl::SplitType::test);
    // NEW
    int n_te_batches = set_NBPP_distributed(n_te_samples, args.batch_size, distr_dataset);

    // Reset batch counter
    dataset.ResetAllBatches(true);
    //dataset.ResetBatch(ecvl::SplitType::test,true);

    // Reset the accumulated loss value
    eddl::reset_loss(model);

    // Auxiliary variables to store the results
    float loss, acc;

    // Validation phase
    float load_time = 0.f;
    float eval_time = 0.f;
      float misc_time = 0.f;
    const auto test_start = std::chrono::high_resolution_clock::now();
    dataset.Start();
    for (int b = 1; b <= n_te_batches; ++b) {
        // Load data
        const auto load_start = std::chrono::high_resolution_clock::now();
        auto [samples, x, y] = dataset.GetBatch();
        const auto load_end = std::chrono::high_resolution_clock::now();
        load_time += std::chrono::duration_cast<std::chrono::microseconds>(load_end - load_start).count();

        // We make sure that the shape of the model is the correct one
        // This may change for the last batch
        model->resize(x.get()->shape[0]);

        // Perform evaluation
        const auto eval_start = std::chrono::high_resolution_clock::now();
        if (args.classifier_output == "sigmoid" && dataset.classes_.size() == 2) {
            Tensor *aux_y = y.get()->select({":", "1"});
            aux_y->reshape_({-1});
            eddl::eval_batch(model,{x.get()},
            {
                aux_y
            });
            delete aux_y;
        } else {
            eddl::eval_batch(model,{x.get()},
            {
                y.get()
            });
        }
        const auto eval_end = std::chrono::high_resolution_clock::now();
        eval_time += std::chrono::duration_cast<std::chrono::microseconds>(eval_end - eval_start).count();

        // Get the current losses and metrics
        loss = eddl::get_losses(model)[0];
        acc = eddl::get_metrics(model)[0];

        if (id == 0) {
            // Show current stats
            std::cout << "[DISTR] Proc " << id << " - ";
            ;
            std::cout << "Testing: Batch " << b << "/" << n_te_batches << ": ";
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "Metrics[ loss=" << loss << ", acc=" << acc << " ]";
            std::cout << " - Timers[ ";
            std::cout << "avg_load_batch=" << (load_time / b) * 1e-6 << "s";
            std::cout << ", avg_eval_batch=" << (eval_time / b) * 1e-6 << "s ]";
            std::cout << " - DataGenerator[ |fifo| = " << dataset.GetQueueSize() << " ]";
            std::cout << std::endl;
        }
    }
    // average params
    avg_float_distributed(&loss);
    avg_float_distributed(&acc);
    int b = n_te_batches;
    // Show current stats
    if (id == 0)
        printf("AVG: Metrics[ loss= %.4f acc= %.4f ]\n", loss, acc);

    dataset.Stop();

    const auto test_end = std::chrono::high_resolution_clock::now();
    const float test_time = std::chrono::duration_cast<std::chrono::microseconds>(test_end - test_start).count();
    if (id == 0)
        std::cout << "Test time elapsed = " << test_time * 1e-6 << "s\n\n";

    // Show the epoch results for each split
    std::cout << std::fixed << std::setprecision(4);
    if (id == 0)
        std::cout << "Test results: loss=" << loss << " - acc=" << acc << "\n\n";

    return TestResults(loss, acc);
}
