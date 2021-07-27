#ifndef _AUGMENTATIONS_HPP_
#define _AUGMENTATIONS_HPP_

#include <memory>
#include <string>
#include <vector>

#include <ecvl/augmentations.h>
#include <ecvl/support_eddl.h>

#include "../utils/utils.hpp"

ecvl::DatasetAugmentations get_augmentations(const Arguments &args);

#endif
