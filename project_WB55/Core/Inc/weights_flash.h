/*
 * weights_flash.h
 *
 *  Created on: Mar 17, 2026
 *      Author: yings
 */

#pragma once
#include <stdint.h>

uint8_t weights_flash_load(void);
uint8_t weights_flash_save(void);

void weights_flash_set_infer_acc(float acc);
float weights_flash_get_infer_acc(void);
