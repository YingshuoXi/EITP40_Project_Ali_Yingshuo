/*
 * save.h
 *
 *  Created on: Feb 19, 2026
 *      Author: yings
 */

#pragma once
#include <stdint.h>
#include <stddef.h>
#include "config.h"


extern double global_x[NN_IN];
extern volatile int8_t global_y;
extern volatile uint8_t global_sample_ready;

void save_store_sample(int8_t y, const uint8_t *data, size_t len);
//void save_clear_ready(void);
