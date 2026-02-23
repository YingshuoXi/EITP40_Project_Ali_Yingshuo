/*
 * nn_ff.h
 *
 *  Created on: Feb 22, 2026
 *      Author: yings
 */

#pragma once
#include <stdint.h>

void nn_ff_init(double lr);

// Training with one sample
double nn_ff_train_one(const double x[50], int8_t y);

// Inference, return probability
double nn_ff_predict(const double x[50]);
