/*
 * protocol_uart.h
 *
 *  Created on: Feb 19, 2026
 *      Author: yings
 */

#pragma once
#include "stm32wbxx_hal.h"
#include <stdint.h>

void protocol_init(UART_HandleTypeDef *hlpuart);	// Initialization
void protocol_start_uart_rx(void);				// Ready to receive data
void protocol_uart_rx_byte(uint8_t b);      	// Message analysis
void protocol_while(void);  					// Used in the while loop
void protocol_set_idle(uint8_t idle);			// Set Idle(1)
void protocol_resume_requesting(void);  		// Request next sample when training is done

void protocol_send_req(void);              		// Send request
void protocol_send_end(void);					// Stop Python code
void protocol_send_results(double loss, double p, uint8_t correct);	// Send Results
