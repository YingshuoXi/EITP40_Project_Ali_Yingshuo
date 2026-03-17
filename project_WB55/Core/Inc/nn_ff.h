/*
 * nn_ff.h
 *
 *  Created on: Feb 22, 2026
 *      Author: yings
 */

#pragma once
#include <stdint.h>
#include <stddef.h>
#include "config.h"

void nn_ff_init(void);

// Training with one sample
float nn_ff_train_one(const float x[NN_FF_IN], int8_t y);

// Inference, return probability
float nn_ff_predict(const float x[NN_FF_IN]);

size_t nn_ff_state_size(void);
void nn_ff_state_export(void *dst);
uint8_t nn_ff_state_import(const void *src, size_t len);
