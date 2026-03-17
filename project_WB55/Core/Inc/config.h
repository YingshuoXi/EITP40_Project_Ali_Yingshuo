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
#define NN_LR_FF		1e-4

#define NN_EPOCHS   	15			// 0 = test only

#define NN_IN			80
#define NN_H1			40
#define NN_H2			20

#define NN_FF_IN		80
#define NN_FF_H1		40
#define NN_FF_H2		20
#define NN_FF_H3		20

#define NN_LOAD_OLD_WEIGHTS_AT_BOOT     1
#define NN_SAVE_NEW_WEIGHTS_AFTER_TRAIN 1


#if (USE_BACKPROP + USE_FF) > 1
	#error "USE_BACKPROP and USE_FF cannot both be 1"
#endif
