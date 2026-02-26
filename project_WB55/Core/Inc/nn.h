/*
 * nn.h
 *
 *  Created on: Feb 21, 2026
 *      Author: yings
 */

#pragma once
#include <stdint.h>
#include "config.h"

void nn_init(void);

// Training with one sample
double nn_train_one(const double x[NN_IN], int8_t y);

// Inference, return probability
double nn_predict(const double x[NN_IN]);
