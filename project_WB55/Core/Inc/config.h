/*
 * config.h
 *
 *  Created on: Feb 25, 2026
 *      Author: yings
 */

#pragma once

#define USE_BACKPROP	1
#define USE_FF			0

#define NN_LR_BACKPROP	1e-3
#define NN_LR_FF		1e-3

#define NN_IN			80
#define NN_H1			20
#define NN_H2			5

#define NN_FF_IN		80
#define NN_FF_H1		20
#define NN_FF_H2		5
#define NN_FF_H3		5

#if (USE_BACKPROP + USE_FF) > 1
	#error "USE_BACKPROP and USE_FF cannot both be 1"
#endif
