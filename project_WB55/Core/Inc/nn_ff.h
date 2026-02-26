/*
 * nn_ff.h
 *
 *  Created on: Feb 22, 2026
 *      Author: yings
 */

#pragma once
#include <stdint.h>
#include "config.h"

void nn_ff_init(void);

// Training with one sample
double nn_ff_train_one(const double x[NN_FF_IN], int8_t y);

// Inference, return probability
double nn_ff_predict(const double x[NN_FF_IN]);
