/*
 * save.c
 *
 *  Created on: Feb 19, 2026
 *      Author: yings
 */


#include "save.h"
#include <string.h>

double global_buf50[50];

void save_store_vector(const uint8_t *data, size_t len)
{
    memcpy(global_buf50, data, len);
}
