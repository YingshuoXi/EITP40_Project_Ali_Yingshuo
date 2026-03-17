/*
 * nn.h
 *
 *  Created on: Feb 21, 2026
 *      Author: yings
 */

#pragma once
#include <stdint.h>
#include <stddef.h>
#include "config.h"

void nn_init(void);

// Training with one sample
float nn_train_one(const float x[NN_IN], int8_t y);

// Inference, return probability
float nn_predict(const float x[NN_IN]);

size_t nn_state_size(void);
void nn_state_export(void *dst);
uint8_t nn_state_import(const void *src, size_t len);
