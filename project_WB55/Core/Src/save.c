/*
 * save.c
 *
 *  Created on: Feb 19, 2026
 *      Author: yings
 */

#include "save.h"
#include <string.h>
#include "config.h"


double global_x[NN_IN];
volatile int8_t global_y = 0;
volatile uint8_t global_sample_ready = 0;

void save_store_sample(int8_t y, const uint8_t *data, size_t len)
{
    memcpy(global_x, data, len);
    global_y = y;
    global_sample_ready = 1;
}

//void save_clear_ready(void)
//{
//	global_sample_ready = 0;
//}
