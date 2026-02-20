/*
 * save.h
 *
 *  Created on: Feb 19, 2026
 *      Author: yings
 */

#pragma once
#include <stdint.h>
#include <stddef.h>

void save_store_vector(const uint8_t *data, size_t len);
extern double global_buf50[50];
