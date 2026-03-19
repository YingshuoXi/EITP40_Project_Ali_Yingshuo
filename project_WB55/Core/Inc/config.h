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

// Train - Validation - Train - Validation - Train ... Validation - Test
// (Train + Validation) = 1 Epoch
#define NN_EPOCHS   	12			// 0 = test only; suggested: 12 epochs

#define NN_IN			80
#define NN_H1			40
#define NN_H2			20

#define NN_FF_IN		80
#define NN_FF_H1		40
#define NN_FF_H2		20
#define NN_FF_H3		20


#define NN_LOAD_OLD_WEIGHTS_AT_BOOT     1	// 0 = Do not load (Random Initialization); 1 = load ONLY IF WEIGHTS_MAGIC and WEIGHTS_VERSION (in weights_flahs.c file) match
#define NN_SAVE_NEW_WEIGHTS_AFTER_TRAIN 1	// 0 = Do not save; 1 = Save weights ONLY IF Validation Acc > Old Recorded one

// NOTE: The flash pages saving weights are reserved and isolated in STM32WB55RGVX_FLASH.ld file (NVM_FLASH: ORIGIN = 0x08078000, LENGTH = 32K)
//       Re-run the project will NOT erase the previous weights, as long as NN_SAVE_NEW_WEIGHTS_AFTER_TRAIN = 0
//   !!! Larger network weights might require larger NVM flash which should be carefully calculated

#if (USE_BACKPROP + USE_FF) > 1
	#error "USE_BACKPROP and USE_FF cannot both be 1"
#endif
