/*
 * nn.h
 *
 *  Created on: Feb 21, 2026
 *      Author: yings
 */

#pragma once
#include <stdint.h>

void nn_init(double lr);

// Training with one sample
double nn_train_one(const double x[50], int8_t y);

// Inference, return probability
double nn_predict(const double x[50]);
